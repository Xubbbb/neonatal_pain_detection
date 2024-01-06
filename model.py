from models import RCA


def generate_model(opt, device):
    assert opt.model in ['RCA']

    model = RCA.RCA(num_classes=opt.num_levels)
    return model.to(device)