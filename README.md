# PyTorch

A comprehensive collection of PyTorch implementations covering fundamental to advanced deep learning concepts. This repository contains practical examples and implementations of various neural network architectures, optimization techniques, and deep learning projects.

## 🎯 Overview

This repository demonstrates various PyTorch implementations from basic tensor operations to complex neural network architectures. Each notebook focuses on specific concepts with hands-on examples using real datasets, making it an ideal resource for learning PyTorch from scratch to advanced applications.

## 📁 Repository Structure

### 📂 Directories

| Directory | Description |
|-----------|-------------|
| `ANN/` | Artificial Neural Network implementations with various configurations |
| `CNN/` | Convolutional Neural Network architectures and transfer learning |
| `RNN/` | Recurrent Neural Network implementations for sequence tasks |
| `Datasets/` | MNIST and Fashion-MNIST datasets for training and testing |

### 📓 Root Level Notebooks

| Notebook | Description |
|----------|-------------|
| `Tensors.ipynb` | Fundamental tensor operations and PyTorch basics |
| `Autograd.ipynb` | Automatic differentiation and gradient computation |
| `Simple_Neural_Network.ipynb` | Building basic neural networks from scratch |
| `NN_Module-1.ipynb` | Introduction to PyTorch's nn.Module framework |
| `NN_Module-2.ipynb` | Advanced nn.Module implementations |
| `Dataset_Dataloader_class.ipynb` | Custom dataset creation and data loading |
| `Optuna.ipynb` | Hyperparameter optimization using Optuna framework |

### 🧠 ANN (Artificial Neural Networks)

| Notebook | Description |
|----------|-------------|
| `ANN(Mnist).ipynb` | Multi-layer perceptron on MNIST dataset |
| `ANN(small_dataset).ipynb` | ANN implementation on small datasets |
| `ANN(Full_dataset).ipynb` | Comprehensive ANN training on complete datasets |
| `ANN(optimization).ipynb` | Various optimization techniques for ANNs |
| `ANN(HyperTunning).ipynb` | Hyperparameter tuning strategies for ANNs |

### 🖼️ CNN (Convolutional Neural Networks)

| Notebook | Description |
|----------|-------------|
| `CNN(Basic).ipynb` | Fundamental CNN architecture and image classification |
| `CNN(Transfer_Learning).ipynb` | Transfer learning implementation using VGG16 |

### 🔄 RNN (Recurrent Neural Networks)

| Notebook | Description |
|----------|-------------|
| `RNN.ipynb` | Question-Answer system using RNN architecture |

### 📊 Datasets

| File | Description |
|------|-------------|
| `fashion-mnist_test.csv` | Fashion-MNIST test dataset in CSV format |
| `t10k-images-idx3-ubyte` | MNIST test images (binary format) |
| `t10k-labels-idx1-ubyte` | MNIST test labels (binary format) |
| `train-images-idx3-ubyte` | MNIST training images (binary format) |
| `train-labels-idx1-ubyte` | MNIST training labels (binary format) |

## 🔧 Features Covered

### Core PyTorch Concepts
- **Tensor Operations**: Basic tensor manipulations and mathematical operations
- **Automatic Differentiation**: Understanding and implementing autograd
- **Neural Network Modules**: Building custom layers and models using nn.Module
- **Data Handling**: Custom datasets, data loaders, and preprocessing

### Neural Network Architectures
- **Artificial Neural Networks**: Multi-layer perceptrons with various configurations
- **Convolutional Neural Networks**: Image classification and computer vision tasks
- **Recurrent Neural Networks**: Sequential data processing and NLP applications

### Advanced Techniques
- **Transfer Learning**: Leveraging pre-trained models (VGG16)
- **Hyperparameter Optimization**: Using Optuna for automated hyperparameter tuning
- **Model Optimization**: Various optimization strategies and techniques
- **Question-Answer Systems**: NLP applications using RNNs

## 🚀 Getting Started

### Prerequisites
```python
torch>=1.9.0
torchvision>=0.10.0
numpy
pandas
matplotlib
seaborn
optuna
jupyter
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Karanmeta/PyTorch.git
cd PyTorch
```

2. Install required packages:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn optuna jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 📈 Learning Path

### Beginner Level
1. **Start with**: `Tensors.ipynb` - Understand PyTorch fundamentals
2. **Gradients**: `Autograd.ipynb` - Learn automatic differentiation
3. **Basic Networks**: `Simple_Neural_Network.ipynb` - Build your first network
4. **Data Handling**: `Dataset_Dataloader_class.ipynb` - Work with datasets

### Intermediate Level
5. **Neural Modules**: `NN_Module-1.ipynb` → `NN_Module-2.ipynb`
6. **ANNs**: Explore the `ANN/` directory notebooks in sequence
7. **CNNs**: `CNN(Basic).ipynb` for computer vision basics

### Advanced Level
8. **Transfer Learning**: `CNN(Transfer_Learning).ipynb`
9. **Sequential Models**: `RNN.ipynb` for sequence processing
10. **Optimization**: `Optuna.ipynb` for hyperparameter tuning

## 🎯 Projects and Applications

### Computer Vision
- MNIST digit classification using ANNs and CNNs
- Fashion-MNIST classification with various architectures
- Transfer learning with VGG16 for image classification

### Natural Language Processing
- Question-Answer system implementation using RNNs
- Sequential data processing techniques

### Optimization and Tuning
- Automated hyperparameter optimization with Optuna
- Comparison of different optimization techniques
- Performance analysis across different architectures

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Karan Mehta** - [Karanmeta](https://github.com/Karanmeta)
- LinkedIn: [Karan Mehta](https://www.linkedin.com/in/karan-mehta-492122333)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Open-source community for datasets and resources
- Contributors and collaborators for their valuable feedback

## 📚 Additional Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch)

---

⭐ Star this repository if you found it helpful for your PyTorch learning journey!