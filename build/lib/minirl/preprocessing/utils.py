import inspect
import types


def _repr_obj(obj, show_modules: bool = False, depth: int = 0) -> str:
    """Return a pretty representation of an object."""

    rep = f"{obj.__class__.__name__} ("
    if show_modules:
        rep = f"{obj.__class__.__module__}.{rep}"
    tab = "\t"

    params = {
        name: getattr(obj, name)
        for name, param in inspect.signature(obj.__init__).parameters.items()  # type: ignore
        if not (
            param.name == "args"
            and param.kind == param.VAR_POSITIONAL
            or param.name == "kwargs"
            and param.kind == param.VAR_KEYWORD
        )
    }

    n_params = 0

    for name, val in params.items():
        n_params += 1

        # Prettify the attribute when applicable
        if isinstance(val, types.FunctionType):
            val = val.__name__
        if isinstance(val, str):
            val = f'"{val}"'
        elif isinstance(val, float):
            val = (
                f"{val:.0e}"
                if (val > 1e5 or (val < 1e-4 and val > 0))
                else f"{val:.6f}".rstrip("0")
            )
        elif isinstance(val, set):
            val = sorted(val)
        elif hasattr(val, "__class__") and "river." in str(type(val)):
            val = _repr_obj(obj=val, show_modules=show_modules, depth=depth + 1)

        rep += f"\n{tab * (depth + 1)}{name}={val}"

    if n_params:
        rep += f"\n{tab * depth}"
    rep += ")"

    return rep.expandtabs(2)
