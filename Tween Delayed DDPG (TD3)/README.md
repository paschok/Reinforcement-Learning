# General Info about TD3-model and steps taken to built it

TD3 model is built on top of Deep Q-Learning and Actor-Critic  concept. It consists of 1 Actor and 2 Critics, which produce different Q-values, 
that are used for gradient ascent to update the parameters of an Actor. These Q-values are learned on past/historic data. 
After all, Q-Learning is an **off-policy** algorithm. 

## The research paper about TD3 can be found with [this link][1]

#### In order to implement the Twin Delayed DDPG, or simply TD3 model the following 15 steps are taken. 

1. Initialize the Experience Replay memory with a size of 20,000 transitions. Each transition is of shape: **(s, s', a, r)**, where
    - s - current state, 
    - s' - next state, 
    - a - action that leads to s'
    - r - reward
 2. Build the NN for Actor model and one NN for Actor target
 3. Build 2 NNs for 2 Critic models and 2 NNs for 2 Critic targets.
  - Overall, we will have 6 neural networks. 2 Actors (model and target) will learn the Policy, whilst 4 Critics (model and target) will learn Q-values
 
 **Now the training process begins:** full episode with first 10,000 actions randomly played and then with the actions played by Actor model. 
 Then repeat the foollowing steps:
 
 4. Sample a batch of transitions (say, 100) from memory. For each element of the batch, make 4 batches of solely *s, s', a, r*
 5. From the next state **s'**, Actor target plays the next action **a**
 6. Add Gaussian noise to this next action **a'** (needed for exploration)
 7. 2 Critic targets take each the couple **(s', a')** as input and return 2 Q-values: **Q<sub>t1</sub>(s',a')** and **Q<sub>t2</sub>(s',a')** as outputs.
 8. Keep the minimum of these two Q-vules: **min(Q<sub>t1</sub>; Q<sub>t2</sub>)** -> approximated value of the next state
 9. Get the final target of the 2 Critic targets, which is **Q<sub>t</sub> = r + y*min(Q<sub>t1</sub>; Q<sub>t2</sub>)**, where *y* is a discount rate
 10. 2 Critic models take each the couple as input and return 2 Q-values **Q<sub>1</sub>(s,a)** and **Q<sub>2</sub>(s,a)** 
 11. Compute the Loss from 2 Critic models: 
  - Critic Loss = MSE(Q<sub>1</sub>(s,a), Q<sub>t</sub>) + MSE(Q<sub>2</sub>(s,a), Q<sub>t</sub>)
 12. Backpropagate this Critic loss and update the params of the 2 Critic models with SGD optimizer
 
 Once every two iterations:
 
 13. Update Actor model by gradient *ascent* on the output of the *first* Critic Model
 14. Update the weights of the Actor target by *polyak averaging* :
  - O<sub>i</sub>' = *tau* * O<sub>i</sub> + (1 - *tau*) *  O<sub>i</sub>', where
    - *tau* - some small number
    -  O<sub>i</sub> - param of an active model
    -  O<sub>i</sub>' - param of an active target before update
 15. Update the weights of Critic target by *polyak averaging* (the same formula, as above)
 
## To train model in different environments, change *env_name* variable, like that:

```python
env_name = "Walker2DBulletEnv-v0" # Humanoid

env_name = "AntBulletEnv-v0" # Ant in 3D-dimension
```


[1]: https://arxiv.org/pdf/1802.09477.pdf
