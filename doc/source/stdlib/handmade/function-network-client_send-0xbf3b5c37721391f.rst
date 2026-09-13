Sends ``size`` bytes from ``data`` to the peer, looping until every byte is out. Returns false, with the socket closed and ``onDisconnect`` called, when the peer went away.
