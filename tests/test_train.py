from model.train import train_model

def test_model_accuracy():
    acc = train_model(capsys)
    print(f"Model accuracy: {acc}")
    captured = capsys.readouterr()
    assert acc > 0.7, f"Model accuracy too low: {acc}. Output:\n{captured.out}"
