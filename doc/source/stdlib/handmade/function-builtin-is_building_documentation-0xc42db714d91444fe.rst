Returns true while the host is compiling a documentation or reflection root.
Macro infrastructure can use this to keep build-machine-specific transforms
and side effects out of generated API documentation.

def is_building_documentation () : bool
