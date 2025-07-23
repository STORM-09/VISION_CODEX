import pickle
from sklearn.svm import SVC

with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)
with open("data/names.pkl", "rb") as f:
    names = pickle.load(f)

model = SVC(kernel='linear', probability=True)
model.fit(embeddings, names)

with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'svm_model.pkl'")