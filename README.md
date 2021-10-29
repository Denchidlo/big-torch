Big-torch library
=================


### Usage:

Big-torch library installation:
```bash
pip install -i https://test.pypi.org/simple/ bigtorch-denissimo
```

Container run:
```bash
docker build --pull -f "container/Dockerfile.prod" -t bigtorch:demo "container"
docker run -ti -rm bigtorch:demo
```

### Context:

#### 1) Big-troch lib

##### It has the following structure:

    big_torch:
    ├── core                    # Mix of different utilities I'm currently unable to gather in one logical module
    ├── layers                  # All possible layers that can form my neural network
    ├── model                   # Model builder
    ├── train                   # Is represented by OptimizationFabric which can easily build complex training algorithms
    ├── preprocessing           # Some useful utilities you may need during your research
    └── remote_client.md        # Allows you to launch models by special configuration files


##### Code example:
```python
from big_torch import model, train, layers, preprocessing

net = models.Model()

net.add_layer(layers.linear.LinearLayer((784, 100), b_initial=0, w_init='xavier_normal'))
net.add_layer(layers.activations.Tanh((100, 100)))
net.add_layer(layers.linear.LinearLayer((100, 100) , b_initial=0, w_init='xavier_normal'))
net.add_layer(layers.activations.Tanh((100, 100)))
net.add_layer(layers.linear.LinearLayer((100, 10) , b_initial=0, w_init='xavier_normal'))
net.add_layer(layers.activations.Softmax((10, 10)))
net.set_loss(layers.loss.CrossEntropy((10, 1)))
net.build()

optimizer = train.OptimizatonFabric(
    generator=train.StochasticBatchGenerator, 
    optimizer=train.GradientDecent,
    optimizer_cfg={'eta': 0.1},
    generator_cfg={'batch_size': 1000}, 
    callbacks=[
        train.EpochInfo(
            validate=True, 
            metric=preprocessing.cross_validation.class_accuracy)
    ]
)

optimizer.train(net, train_x, train_y, x_val=test_x, y_val=text_y, min_eps=0.0000005, max_iter=2000)
```

#### 2) Container:

This folder will provide you easy to launch docker container (Dockerfile.prod). You'll just need to specify model structure and credentials to neptune.ai in ***config*** folder. 
Note that Dockerfile.build is used in my CI and you don't need to run it manualy. 

> Results you can see on neptune tracker. [See.](https://app.neptune.ai/denissimo/MNIST-Big-Torch/)
