
def _set_config_attributes(instance, init_args_dict: dict, defaults_dict: dict) -> None:
    for var_name, default_value in defaults_dict.items():
        value = init_args_dict.get(var_name)
        setattr(instance, var_name, value if value is not None else default_value)
        