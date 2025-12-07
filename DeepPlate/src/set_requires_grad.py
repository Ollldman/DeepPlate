def set_requires_grad(model, unfreeze_pattern, verbose=False):
    # Если пустая строка - замораживаем все
    if len(unfreeze_pattern) == 0:
        for _, param in model.named_parameters():
            param.requires_grad = False
        return

    # разбиваем все слои
    pattern = unfreeze_pattern.split("|")

    # Проходим по всем слоям и ищем совпадения любого
    # слоя из `pattern` с текущим именем слоя
    for name, param in model.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False