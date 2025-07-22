from model.train import train_model

def test_model_accuracy():
    acc = train_model()
    print(f"Model accuracy: {acc}")
    assert acc > 0.7, f"Model accuracy too low: {acc}"
