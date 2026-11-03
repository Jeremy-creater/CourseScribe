"""Show or change the default output device. A test helper, always run as a child process.

``--show`` prints, as JSON, the default output device ID for each role.
``--set ID [--role ROLE]`` makes ID the default for that role (all three when no role is given).

Changing uses ``IPolicyConfig::SetDefaultEndpoint``: not documented, but what the volume mixer
and tools such as SoundVolumeView use. It changes the current user's setting, the same as
picking a device from the taskbar, so it needs no administrator rights. Running it in its own
process means a COM mistake can only crash this process, never the test run.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from ctypes import POINTER, byref, c_uint, c_void_p, c_wchar_p

ROLES = {"console": 0, "multimedia": 1, "communications": 2}
_RENDER = 0
_CLSCTX_ALL = 23
_COINIT_APARTMENTTHREADED = 2

_CLSID_ENUMERATOR = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
_IID_ENUMERATOR = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
_CLSID_POLICY_CONFIG = "{870AF99C-171D-4F9E-AF0D-E63DF40C2BC9}"
_IID_POLICY_CONFIG = "{F8679F50-850A-41CF-9C72-430F290290C8}"

# Places in each interface's table of methods (the first three are IUnknown's).
_RELEASE = 2
_GET_DEFAULT_AUDIO_ENDPOINT = 4  # IMMDeviceEnumerator
_GET_DEVICE = 5  # IMMDeviceEnumerator
_GET_ID = 5  # IMMDevice
_GET_STATE = 6  # IMMDevice
_DEVICE_STATE_ACTIVE = 1
_SET_DEFAULT_ENDPOINT = 13  # IPolicyConfig

_ole32 = ctypes.OleDLL("ole32")  # calls that return HRESULT: failures raise OSError
_memory = ctypes.WinDLL("ole32")  # CoTaskMemFree returns nothing


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid(text: str) -> _GUID:
    guid = _GUID()
    _ole32.CLSIDFromString(c_wchar_p(text), byref(guid))
    return guid


def _method(pointer: c_void_p, index: int, *argtypes):
    """The method at ``index`` of a COM object, with HRESULT failures raised as OSError."""
    table = ctypes.cast(pointer, POINTER(POINTER(c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)
    function = prototype(table[index])
    return lambda *args: function(pointer, *args)


def _release(pointer: c_void_p) -> None:
    if pointer:
        ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(
            ctypes.cast(pointer, POINTER(POINTER(c_void_p))).contents[_RELEASE]
        )(pointer)


def _create(clsid: str, iid: str) -> c_void_p:
    pointer = c_void_p()
    _ole32.CoCreateInstance(byref(_guid(clsid)), None, _CLSCTX_ALL, byref(_guid(iid)), byref(pointer))
    return pointer


def show() -> dict[str, str]:
    enumerator = _create(_CLSID_ENUMERATOR, _IID_ENUMERATOR)
    try:
        defaults = {}
        for role, number in ROLES.items():
            device = c_void_p()
            _method(enumerator, _GET_DEFAULT_AUDIO_ENDPOINT, c_uint, c_uint, POINTER(c_void_p))(
                _RENDER, number, byref(device)
            )
            try:
                text = c_void_p()
                _method(device, _GET_ID, POINTER(c_void_p))(byref(text))
                defaults[role] = ctypes.wstring_at(text.value)
                _memory.CoTaskMemFree(text)
            finally:
                _release(device)
        return defaults
    finally:
        _release(enumerator)


def state(device_id: str) -> int:
    """The device's state (1 = enabled and present). Raises OSError for an unknown ID."""
    enumerator = _create(_CLSID_ENUMERATOR, _IID_ENUMERATOR)
    try:
        device = c_void_p()
        _method(enumerator, _GET_DEVICE, c_wchar_p, POINTER(c_void_p))(device_id, byref(device))
        try:
            value = ctypes.c_uint()
            _method(device, _GET_STATE, POINTER(ctypes.c_uint))(byref(value))
            return value.value
        finally:
            _release(device)
    finally:
        _release(enumerator)


def set_default(device_id: str, roles: list[str]) -> None:
    current = state(device_id)
    if current != _DEVICE_STATE_ACTIVE:
        raise SystemExit(f"refusing {device_id}: its state is {current}, not enabled")
    policy = _create(_CLSID_POLICY_CONFIG, _IID_POLICY_CONFIG)
    try:
        for role in roles:
            _method(policy, _SET_DEFAULT_ENDPOINT, c_wchar_p, c_uint)(device_id, ROLES[role])
    finally:
        _release(policy)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--show", action="store_true")
    action.add_argument("--set", metavar="ID")
    parser.add_argument("--role", choices=sorted(ROLES))
    options = parser.parse_args(argv)
    _ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
    if options.set:
        set_default(options.set, [options.role] if options.role else list(ROLES))
    print(json.dumps(show()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
