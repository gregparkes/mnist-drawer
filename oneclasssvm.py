"""basic training script of One class svm"""

import torchvision.datasets as datasets
from sklearn.linear_model import SGDOneClassSVM
import json

mnist_train = datasets.MNIST(root = "./data", train=True, download=False)

X = ((mnist_train.data - 127.5) / 127.5).numpy()
Xnew = X.reshape(60000, 28*28)

model = SGDOneClassSVM()
model.fit(Xnew)

print(model.predict(Xnew))

# save as a special format - ONNX does not work.
json_data = {
    "coef_": model.coef_.tolist(),
    "offset_": model.offset_.tolist(),
    "n_iter_": model.n_iter_,
    "t_": int(model.t_),
    "n_features_in_": int(model.n_features_in_),
}

with open("ocsvm.json","wt") as f:
    json.dump(json_data, f)
