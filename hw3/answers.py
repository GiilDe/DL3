r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

First of all, the reason to divide the text into sequences is that our goal is to predict the next character based on its
relative position in the last few wards. It doesn't make sense to try to predict last character based on all 
the text that comes before, because most of the text doesn't effect the last character.   

Beyond that, if we train on the whole text, our model dims are going to be huge, it will have a lot of parameters, 
but the model is not going to be trainable on just one sample (=the whole text is one sample in this scenario). 

"""

part1_q2 = r"""
**Your answer:**



"""

part1_q3 = r"""
**Your answer:**

The order between the batches is important, because we want to preserve the order of wards and sentences in the text that 
our model trains on. That because our model learns to generate characters based on their locations and relations to other 
wards in a specific order, so if we change the order we may "confuse" the model and the training will be less effective.
"""

part1_q4 = r"""
**Your answer:**

1.  As we saw before, a low value of temperature will result in less uniform distributions and vice-versa.
During the training, we are less sure about the next char to generate, therefore we would like to "explore" the options,
and allow the model to pick the next char which is no  
On the other hand, when sampling we would prefer to control the distributions and make them less uniform to increase the 
chance of sampling the char(s) with the highest scores compared to the others.  that's because our model is already trained,
so the char with the highest score is probably the best char to generate now. 

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=5,
        h_dim=64, z_dim=64, x_sigma2=0.9,
        learn_rate=0.001, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


