Launches a detached subprocess directly from an argument array without invoking
a shell. ``args[0]`` is the executable and the remaining elements are passed as
individual arguments, so spaces and shell metacharacters are preserved
literally. The child inherits the current working directory, environment, and
standard output/error; standard input is disconnected.

Returns ``true`` once the executable has been started and does not wait for it
to exit. Returns ``false`` if the process could not be created or the executable
could not be started.
