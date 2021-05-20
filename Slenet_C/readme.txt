slenet_params.h - файл, содержит заранее натренированные параметры.
Miras_Bakytbek_FinalMnistTest.cu - в этом файле выполняется forward pass с помощью натренированных параметров на 10000 тестовы снимков с MNIST Dataset. Параллельная чать выполнялась с помощью CUDA library.
Папка data сожержит 10000 тестовых снимков MNIST dataset в байтовом формате.

Все слои были прописаны вручную с помощью параллельного программирования на C, с использованием CUDA (GPU).
За основу была взята s-Lenet архитектура. Модель состоит из 1 CNN layer, 1 Subsampling(Pooling) layer and 1 Fully-connceted layer.