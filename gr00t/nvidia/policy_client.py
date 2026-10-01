# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from enum import Enum
import functools
import io
import json
from typing import Any

import zmq
import msgpack
import msgpack_numpy as mnp
import numpy as np

from .modality_config import ModalityConfig

class GROOTBasePolicy(ABC):
    """Abstract base class for robotic control policies.

    This class defines the interface that all policies must implement, including
    methods for action computation, input/output validation, and state management.

    Subclasses must implement:
        - check_observation(): Validate observation format
        - check_action(): Validate action format
        - _get_action(): Core action computation logic
        - reset(): Reset policy to initial state
    """

    def __init__(self, *, strict: bool = True):
        self.strict = strict

    @abstractmethod
    def check_observation(self, observation: dict[str, Any]) -> None:
        """Check if the observation is valid.

        Args:
            observation: Dictionary containing the current state/observation of the environment

        Raises:
            AssertionError: If the observation is invalid.
        """
        pass

    @abstractmethod
    def check_action(self, action: dict[str, Any]) -> None:
        """Check if the action is valid.

        Args:
            action: Dictionary containing the action to be executed

        Raises:
            AssertionError: If the action is invalid.
        """
        pass

    @abstractmethod
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute and return the next action based on current observation.

        This method should be overridden by subclasses to implement policy-specific
        action computation. Input validation is handled by the public get_action() method.

        Args:
            observation: Dictionary containing the current state/observation
            options: Optional configuration dict for action computation

        Returns:
            Tuple of (action, info):
                - action: Dictionary containing the action to be executed
                - info: Dictionary containing additional metadata (e.g., confidence scores)
        """
        pass

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute and return the next action based on current observation with validation.

        This is the main public interface. It validates the observation, calls
        the internal _get_action(), and validates the resulting action.

        Args:
            observation: Dictionary containing the current state/observation
            options: Optional configuration dict for action computation

        Returns:
            Tuple of (action, info):
                - action: Dictionary containing the validated action
                - info: Dictionary containing additional metadata

        Raises:
            AssertionError/ValueError: If observation or action validation fails
        """
        if self.strict:
            self.check_observation(observation)
        action, info = self._get_action(observation, options)
        if self.strict:
            self.check_action(action)
        return action, info

    @abstractmethod
    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        pass

class MsgSerializer:
    """msgpack_numpy serializer with a hard ``allow_pickle=False`` boundary.

    Implementation note: msgpack_numpy's ``Packer``/``Unpacker`` wire any
    user-provided ``default``/``object_hook`` *behind* their own
    ``mnp.encode``/``mnp.decode`` (``functools.partial(encode, chain=user_fn)``).
    That means an object-dtype ndarray is serialised by ``mnp.encode`` (and a
    forged ``{nd: True, kind: 'O', ...}`` payload is fed to ``pickle.loads`` by
    ``mnp.decode``) *before* a hook installed via ``mnp.packb``/``mnp.unpackb``
    can intervene. We therefore drive ``msgpack`` directly and chain
    ``_safe_{encode,decode} → mnp.{encode,decode} → custom`` ourselves, so the
    refusal runs first.
    """

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        default = functools.partial(MsgSerializer._safe_encode, chain=MsgSerializer._encode_custom)
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        object_hook = functools.partial(
            MsgSerializer._safe_decode, chain=MsgSerializer._decode_custom
        )
        return msgpack.unpackb(data, object_hook=object_hook, raw=False)

    @staticmethod
    def _safe_encode(obj, chain=None):
        # Refuse object-dtype ndarrays before mnp.encode would emit a
        # ``{nd: True, kind: 'O', data: pickle.dumps(arr)}`` envelope, which
        # silently re-enables the arbitrary-code surface that the previous
        # ``np.save(..., allow_pickle=False)`` path explicitly forbade.
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == "O":
                raise TypeError(
                    f"Refusing to encode object-dtype ndarray (shape={obj.shape}); "
                    f"msgpack_numpy would invoke pickle. Convert to a concrete "
                    f"numeric dtype before sending."
                )
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj, chain=None):
        if isinstance(obj, dict):
            marker = obj.get("__ndarray_class__", obj.get(b"__ndarray_class__"))
            if marker:
                payload = obj.get("as_npy", obj.get(b"as_npy"))
                if payload is None:
                    raise ValueError(
                        "Malformed ndarray payload: marker present but 'as_npy' missing"
                    )
                return np.load(io.BytesIO(payload), allow_pickle=False)

        # Refuse object-dtype ndarray payloads before mnp.decode would call
        # ``pickle.loads`` on attacker-controlled bytes. Check both bytes- and
        # str-encoded keys, and accept any truthy ``nd`` value (not just
        # boolean ``True``) so a forged ``{nd: 1, kind: 'O', ...}`` payload
        # can't sidestep the guard via msgpack's int-vs-bool type codes.
        # msgpack_numpy 0.4.8's own check is ``obj[b'nd'] is True``, so the
        # described variants don't actually reach pickle.loads today, but
        # MsgSerializer enforces the contract at this boundary regardless of
        # mnp's wire / identity-check conventions.
        if isinstance(obj, dict):
            nd_val = obj.get(b"nd", obj.get("nd"))
            kind_val = obj.get(b"kind", obj.get("kind"))
            if nd_val and kind_val in (b"O", "O"):
                raise ValueError(
                    "Refusing to decode object-dtype ndarray payload (pickle-bearing); "
                    "the allow_pickle=False contract is enforced by MsgSerializer."
                )
        return mnp.decode(obj, chain=chain)

    @staticmethod
    def _encode_custom(obj):
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig__": True, "as_json": to_json_serializable(obj)}
        return obj

    @staticmethod
    def _decode_custom(obj):
        if not isinstance(obj, dict):
            return obj
        # If the ModalityConfig marker is present but 'as_json' is missing,
        # raise instead of returning a half-broken dict.
        has_modality_marker = (
            "__ModalityConfig__" in obj
            or b"__ModalityConfig__" in obj
            or "__ModalityConfig_class__" in obj
            or b"__ModalityConfig_class__" in obj
        )
        if has_modality_marker:
            key = next((k for k in ("as_json", b"as_json") if k in obj), None)
            if key is None:
                raise ValueError(
                    f"Malformed ModalityConfig payload: marker present but "
                    f"'as_json' missing. keys={sorted(repr(k) for k in obj.keys())}"
                )
            payload = obj[key]
            if isinstance(payload, bytes):
                payload = payload.decode()
            if isinstance(payload, str):
                payload = json.loads(payload)
            return ModalityConfig(**payload)
        return obj

class PolicyClient(GROOTBasePolicy):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
        strict: bool = False,
    ):
        super().__init__(strict=strict)
        self._closed = False
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            # Timeout — REQ socket is now in an invalid state (waiting for a
            # reply that will never arrive).  Recreate it so the next call can
            # send again, then re-raise so the caller knows this request failed.
            self._init_socket()
            raise
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure we are running the correct policy server.")
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def close(self) -> None:
        """Release the REQ socket and ZMQ context. Idempotent."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        socket = getattr(self, "socket", None)
        if socket is not None:
            try:
                socket.close(linger=0)
            except Exception:
                pass
        context = getattr(self, "context", None)
        if context is not None:
            try:
                context.term()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        # Best-effort GC fallback. ``__del__`` can fire during interpreter
        # shutdown after module-level names (``zmq``, our own attributes)
        # have been torn down; raising here is just noise on stderr.
        try:
            self.close()
        except Exception:
            pass

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def check_observation(self, observation: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_observation is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )

    def check_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_action is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )

def to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert dataclasses and numpy arrays to JSON-serializable format.

    Args:
        obj: Object to convert (can be dataclass, numpy array, dict, list, etc.)

    Returns:
        JSON-serializable representation of the object
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        # Convert dataclass to dict, then recursively process the dict
        return to_json_serializable(asdict(obj))
    elif isinstance(obj, np.ndarray):
        # Convert numpy array to list
        return obj.tolist()
    elif isinstance(obj, np.integer):
        # Convert numpy integers to Python int
        return int(obj)
    elif isinstance(obj, np.floating):
        # Convert numpy floats to Python float
        return float(obj)
    elif isinstance(obj, np.bool_):
        # Convert numpy bool to Python bool
        return bool(obj)
    elif isinstance(obj, dict):
        # Recursively process dictionary values
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively process list/tuple elements
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        # Convert set to list
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Already JSON-serializable
        return obj
    elif isinstance(obj, Enum):
        return obj.name
    else:
        # For other types, try to convert to string as fallback
        # You might want to handle specific types differently
        return str(obj)
