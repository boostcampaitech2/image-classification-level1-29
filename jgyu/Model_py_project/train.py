import config
from config import *
import model

gyudel, loss_fn, optim = model.get_model()

def train_batch(img, label, _model, optim, loss_fn):
    _model.train()
    pred = _model(img)
    batch_loss = loss_fn(pred, label)
    
    batch_loss.backward()
    optim.step()
    optim.zero_grad()

    return batch_loss.item()

for epoch in range(EPOCH):
    print(f" epoch {epoch + 1}/10")

    for ix, batch in enumerate(iter(model.data_loader)):
        img, label = batch
        batch_loss = train_batch(img.cuda(), label.cuda().long(), gyudel, optim, loss_fn)


all_predictions = []
gyudel.eval()
for images in model.test_data_loader:
    with torch.no_grad():
        images = images.to(device)
        pred = gyudel(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
model.submission['ans'] = all_predictions

model.submission.to_csv(os.path.join(TEST_DIR, 'submission.csv'), index=False)

print('test inference is done!')