# Face Similarity Search with FaceNet

This project demonstrates **face similarity search using FaceNet embeddings**. It compares two different approaches for vector similarity search:

1. **In-memory vector search using FAISS**
2. **Vector database search using Qdrant**

The notebook generates embeddings for faces and performs nearest neighbor search to find visually similar faces.

---

# Overview

Face recognition systems typically follow this pipeline:

```
Image
  ↓
Face Detection
  ↓
Face Embedding (FaceNet)
  ↓
Vector Similarity Search
  ↓
Top Matching Faces
```

In this project:

* **FaceNet** generates a 512-dimensional embedding for each face.
* These embeddings are stored in either:

  * an **in-memory FAISS index**
  * a **vector database (Qdrant)**

---

# Dataset

The notebook uses the **Pins Face Recognition dataset**.

Dataset:

https://www.kaggle.com/datasets/hereisburak/pins-face-recognition

Dataset details:

* ~17,000 images
* 105 people
* Each person has multiple images

The dataset is automatically downloaded in the notebook using Kaggle.

---

# Project Structure

```
facenet-notebook/
│
├─ facenet.ipynb
├─ README.md
```

---

# Installation

Install dependencies:

```
pip install facenet-pytorch
pip install faiss-cpu
pip install qdrant-client
pip install tqdm matplotlib
```

---

# Running the Notebook

Open the notebook in **Google Colab** or **Jupyter Notebook**.

Steps:

1. Install dependencies
2. Load FaceNet model
3. Generate embeddings for dataset images
4. Build FAISS index
5. Run similarity search
6. Store vectors in Qdrant
7. Perform vector DB search

---

# Section A: In-Memory Search (FAISS)

FAISS stores vectors in memory and performs similarity search.

Advantages:

* Extremely fast
* Simple setup
* Good for datasets up to millions of vectors

Example:

```
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(512)
index.add(embeddings)

distances, indices = index.search(query, k)
```

---

# Section B: Vector Database Search (Qdrant)

Instead of managing the index manually, vectors can be stored in a vector database.

Advantages:

* Persistent storage
* Metadata filtering
* Horizontal scalability
* API support

Example:

```
client.query_points(
    collection_name="faces",
    query=query_vector,
    limit=5
)
```

---

# Example Result

Querying an image of **Bill Gates** returns the closest matches:

```
score: 0.97  Bill Gates
score: 0.95  Bill Gates
score: 0.92  Bill Gates
score: 0.91  Bill Gates
score: 0.89  Bill Gates
```

---

# FAISS vs Vector Database

| Feature            | FAISS          | Qdrant        |
| ------------------ | -------------- | ------------- |
| Speed              | Very fast      | Fast          |
| Persistence        | Manual         | Built-in      |
| Scaling            | Single machine | Distributed   |
| Metadata filtering | No             | Yes           |
| Ease of use        | Simple         | More features |

---

# Future Improvements

Possible extensions:

* Use **ArcFace embeddings instead of FaceNet**
* Use **FAISS IVF index for large-scale search**
* Add **real-time webcam inference**
* Scale to **millions of face embeddings**

---

# License

This project is for educational and research purposes.
