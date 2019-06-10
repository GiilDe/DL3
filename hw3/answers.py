r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001 #makes it become infinity, not good
    # TODO: Tweak the parameters to generate a literary masterpiece.
    start_seq = "ACT I. SEEVANT. And so not me the sing a crown thy and for the sense for my means to store the take"
    temperature = .5
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

The generated text seems to have memory that is longer then the sequence length because we pass all the hidden states
of the network at the end of the sequence to the next word to generate.  We do not want to generate words which have no 
connection to the words we already generated. Instead, we want the generated text to complete and add to the given
sequence. And that is possible due to the hidden states that we pass on. 

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
and allow the model to pick the next char with some measure of randomness.
On the other hand, when sampling, we would prefer to control the distributions and make them less uniform to increase the 
chance of sampling the char(s) with the highest scores compared to the others.  that's because our model is already trained,
so the char with the highest score is probably the best char to generate now. 

2.  When the temperature is very high, we can get surprising characters in non-related locations, because the distribution
we sample characters from is being more and more uniform as the temperature increases. 

3.  When the temperature is very low, the number of spelling-mistake decreases, and we see that the text has many 
repetitions of certain phrases, because we are close to a deterministic predictor.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['h_dim'] = 1024
    hypers['z_dim'] = 3
    hypers['x_sigma2'] = 3
    hypers['learn_rate'] = 0.0003
    hypers['betas'] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The  𝜎2  hyperparameter effect the spread of the generated samples around the approximated mean value of the instance space.
A low value of 𝜎2 will result in samples which are mostly similar to each other.
A high value of 𝜎2 will result in a larger variety of samples, less similar to each other, and the model
will be considered more creative. 

"""

part2_q2 = r"""
**Your answer:**

1. reconstruction loss: we want to measure the distance between the actual data, and the reconstructed data (which went
through the encode and decode process). Our model goal therefore is to maximize the likelihood of generated instances 
from over the entire latent space. 
Notice that is equal to minimizing the expression -𝔼xlog𝑝(X). This expression is intractable, so we use the posterior 
instead, and the final reconstruction loss expression is therefore : 𝔼z∼𝑞𝛼(log𝑝𝛽(X|z))
KL divergence loss:  Intuitively, we can think of KL divergence loss as the statistical measure of how one distribution
(in our case, 𝑞(Z|X)) differs from another (𝑝(Z)).  in order to be able to calculate the data loss, we are using the 
posterior 𝑞(Z|X) instead of the prior distribution 𝑝(Z), as we mentioned earlier. That makes the loss calculation inaccurate. 
The purpose of KL divergence loss is to measure this inaccuracy, so we can minimize it at well as we try to minimize
the actual data loss.

2. By minimizing the KL divergence loss term, we make the latent-space distribution and the instance-space distribution 
statistically independent (by definition :  𝑞(Z|X) = 𝑝(Z)).  

3. The benefit is that we can tune both distributions without effecting each other. the distributions parameters are
tuned by different nets, which we train separately. So the distributions being I.I.D makes our calculations more stable 
and precise. 

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
    hypers['batch_size'] = 32
    hypers['z_dim'] = 10
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.3
    hypers['discriminator_optimizer']['type']='Adam'
    hypers['discriminator_optimizer']['weight_decay']=0.002
    hypers['discriminator_optimizer']['betas']=(0.5,0.999)
    hypers['discriminator_optimizer']['lr']=0.0002
    hypers['generator_optimizer']['type']='Adam'
    hypers['generator_optimizer']['weight_decay']=0.002
    hypers['generator_optimizer']['betas']=(0.4,0.999)
    hypers['generator_optimizer']['lr']=0.0002


    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
During training, we train both models - the discriminator and the generator simultaneously. 
That leads to that we are sampling from our GAN in two different situations:
One is, when we want to calculate the generator loss. we apply the forward path on the sampled objects in the sampling
function. Later, when we calculate the loss we will have them to be in the the generator's computation graph. In this 
case, we maintain gradients when sampling from the GAN.
The second is, when we calculate the loss over the discriminator model. 
in this case we are not interested in the gradients when sampling from GAN, because we use sampling just a pre-preparation 
for applying the discriminator path.  In this case, we do not want to maintain gradients, these are standalone tensors.

"""

part3_q2 = r"""
**Your answer:**

1. no. If our discriminator is not trained well, and is doing a bad job in classifying real and fake images, then the 
generator loss function can decrease below some threshold, although the generator model is not doing a good job as well. 
The purpose is, that the generator loss function can decrease in two scenario: when the generator is getting better, 
or when the discriminator is getting worse.  We want to avoid the second scenario. 
2.  It means that the generator is getting better, produces generated samples that look more real, and in the same time
the discriminator is also getting better, distinguishing between fake and real samples. 

"""

part3_q3 = r"""
**Your answer:**

In general, we can determine the results we got when generating images with the GAN were better than VAE results.
The cause for that may be the loss function we used in those models: in VAE model, we used point-wise loss between original 
samples and reconstructed images. On the other hand, in GAN we used much more complicated loss function.  the GAN loss is
based on a net that supposed to distinguish between real and fake images, this net also improves through the time,
and forcing the generator to keep improving with it. 
Another difference, is about the variety of images models are generating. We observe that the GAN model produced more 
creative images, not all images looks the same, there are different poses and angels of Bush. The reason for it may be
that the GAN model is not trying to get closer to one image at a time, but trying to full the discriminator, so it has to
be unique every time, else the discriminator will learn how to distinguish the fake images. 

"""

# ==============


