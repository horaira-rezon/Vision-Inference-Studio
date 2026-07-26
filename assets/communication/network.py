"""
Network Module: talks to a Raspberry Pi in the field over the network
(the "Remote Setup" option). Intentionally unimplemented for now -- the
Setup screen shows an "underdeveloped" message instead of calling this.

When this gets built, it will implement the same read()/has_depth
interface as camera/base.py's CameraSource, so the GUI's frame loop won't
need to change at all -- it will just receive a RemotePiSource instead of
a WebcamSource or RealSenseSource.
"""


class RemotePiLink:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Remote (Raspberry Pi) mode is not implemented yet.")