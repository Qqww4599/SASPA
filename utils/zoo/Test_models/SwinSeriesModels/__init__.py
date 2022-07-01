from segmentation_models_pytorch import create_model


def modelFactory(modelname: str):
    if modelname == 'modelname1':
        return create_model(modelname)
    else:
        raise ValueError(f'{modelname} not available model!!')