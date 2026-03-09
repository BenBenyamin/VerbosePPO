"""
-----------------------------------------------------------
-----------------------------------------------------------
      ___                ___              ___     
     /  /\              /  /\            /  /\    
    /  /::\            /  /::\          /  /::\   
   /  /:/\:\          /  /:/\:\        /  /:/\:\  
  /  /:/~/:/         /  /:/~/:/       /  /:/  \:\ 
 /__/:/ /:/         /__/:/ /:/       /__/:/ \__\:\
 \  \:\/:/          \  \:\/:/        \  \:\ /  /:/
  \  \::/            \  \::/          \  \:\  /:/ 
   \  \:\             \  \:\           \  \:\/:/  
    \  \:\             \  \:\           \  \::/   
     \__\/ ROXIMAL      \__\/ OLICY      \__\/ PTIMIZATION

-----------------------------------------------------------
Ben Benyamin
-----------------------------------------------------------
03/02/2026
-----------------------------------------------------------
Version 1 
-----------------------------------------------------------

Reinforcement Learning Jargon:

π   : The policy.
θ   : The policy weights / parameters.
a   : The action.
s   : The state
𝔼[x]: The expected value of x. Practically the mean of x.
Gt  : The return, Accumlated discounted reward (starting from time t)
      
      Gt = Σ γ^k *r_{t+k+1)
      where k = 0 -> ∞ , k ∈ ℕ

[The value function , V]
The value is the expected accumlated discounted reward given a given state s:
    V(s) = 𝔼[G|s] = 𝔼[(]Σ(γ^k *r | s))]
    γ = discount factor for future rewards.

[The Q-value function,Q]

The Q value is the expected accumlated discounted reward of taking a specific action a give state s.

Q(s,a) = 𝔼[Gt|s,a] = 𝔼[Σ(γ^k *r | s , a)] 

[Difference between Q and V]

V  = f(s)  measures how good the state is on average, 
Q = f(s,a) measures how good a *specific action* given the state.

[The advantage function, A]
    
The difference between the Q value and the value. 
The advantage measures how much better or worse a specific action is compared to 
the average expected value of the state.

A(s,a) = Q(s,a) - V(s)

A > 0 : The action is better than the average.
A < 0 : The action is worse than the average.

[Temporal Distance Error, δ_t]

The Temporal Difference (TD) error means how much the reward plus the next
value differs from the current value estimate.

δ_t = r_t + γ V(s_{t+1}) − V(s_t)

"""
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as D
import gymnasium as gym

class ActorCritic(nn.Module):
    """
    Adapted from cleanRL : https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

    The base [policy , π_θ] for PPO. This is essentially the PPO's core.
    This is a simple MLP, but there is also an option to pass in a custom network.

    In the default case , the network structure looks like this:
    Input  : Observation 
    Output : Action probability distribution , value
    
    Actor:
        The actor's job is to choose the best action given the observation. If not specified:
        [observation]               (state_dim)
            → [hidden layer]        (hidden_size)
            → [hidden layer]        (hidden_size)
            → [action distribution] (action_dim)
    Critic:
        The critic's job is to estimate the value of the current state given the observation.
        [observation]        (state_dim)
            → [hidden layer] (hidden_size)
            → [hidden layer] (hidden_size)
            → [value]        (1)

    Args:
        state_dim (int): Dimension of the input observation.
        action_dim (int): Dimension of the output action.
        hidden_size (int): The size of the hidden layer. Default is 64 (taken from cleanRL)
        actor (nn.Module, optional): Optional custom actor network. If None then the one from cleanRL is used.
        continuous (bool, optional): Does the environment have continuous actions. Defaults to False.
        init_weights(bool, optional): Whether to weight initialization or not. Defaults to True.
        actor_std (float, optional): Standard deviation for continuous actor distribution. Defaults to 0.01.
        critic_std (float, optional): Initialization scale for critic output layers. Defaults to 1.0.
        hidden_std (float, optional): Initialization scale for hidden layers. Defaults to √2.
        bias_const (float, optional): Constant value to initialize network biases with. Defaults to 0.0.
    """
    def __init__(
                self,
                state_dim:int, 
                action_dim:int,
                hidden_size:int=64,
                actor:nn.Module = None,
                continuous:bool=False,
                actor_std:float=0.01,
                init_weights:bool= True,
                critic_std:float=1.0,
                hidden_std:float = np.sqrt(2),
                bias_const:float=0.0,
                ):
        super().__init__()

        self.continuous = continuous

        if self.continuous:
        # For continuous actions, create a learnable log-standard-deviation vector.
        # This defines the standard deviation (σ) of the Gaussian policy.
        # The actor outputs the mean μ, and together μ + σ·N(0,1) defines the action distribution.
        # Defined as log(σ) because exp(log(sigma)) > 0 , and is numerically stable.
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        if actor is None:
        # Shared or separate networks
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, action_dim)
            )
        else:
            self.actor = actor

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        if init_weights:
            self._init_weights(
                                actor_std = actor_std,
                                critic_std= critic_std,
                                hidden_std=hidden_std,
                                bias_const = bias_const,
                                )

    def _init_weights(
                    self, 
                    actor_std:float=0.01,
                    critic_std:float=1.0,
                    hidden_std:float = np.sqrt(2),
                    bias_const:float=0.0
                    ):
        """
        Initialize network weights. Proper initialization helps PPO start with reasonable value estimates
        and stable action distributions (preserves signal variance across layers).
        Applies orthogonal initialization to all linear layers.
        Steps:
        1. Initialize A: a_ij ~ N(0, 1)
        2. QR decomposition (https://en.wikipedia.org/wiki/QR_decomposition):
            A = Q R
            Q: Orthogonal matrix (Q Q^T = I)
            R: Upper triangular matrix
        3. Set the layer weights (W) = Q * σ
        This approximately achieves w_ij ~ N(0, σ^2) (Gaussian distribution) and W W^T = I (orthogonality)

        For the hidden layers, σ defaults to √2 because, according to He et alia 2015 (https://arxiv.org/pdf/1502.01852), 
        ReLU halves the variance, so to mitigate that we set σ^2 = 2.
        CleanRL uses √2 in spite of using tanh (which does not half the variance) 
        as the activation function; thus, so does this implementation.

        Args:
            actor_std (float, optional): Standard deviation for continuous actor distribution. Defaults to 0.01.
            critic_std (float, optional): Initialization scale for critic output layers. Defaults to 1.0.
            hidden_std (float, optional): Initialization scale for hidden layers. Defaults to √2.
            bias_const (float, optional): Constant value to initialize network biases with. Defaults to 0.0.
        """

        actor_linear_layers = [layer for layer in self.actor.modules() if isinstance(layer, nn.Linear)]
        # Actor layers
        for i, layer in enumerate(actor_linear_layers):
            if isinstance(layer, nn.Linear):
                std = actor_std if i == len(actor_linear_layers)-1 else hidden_std
                nn.init.orthogonal_(layer.weight, gain=std)
                nn.init.constant_(layer.bias, bias_const)

        # Critic layers
        for i, layer in enumerate(self.critic):
            if isinstance(layer, nn.Linear):
                std = critic_std if i == len(self.critic)-1 else hidden_std
                nn.init.orthogonal_(layer.weight, gain=std)
                nn.init.constant_(layer.bias, bias_const)
    
    def forward(self,obs:torch.tensor):

        """
        Forward pass of the ActorCritic policy.

        Computes:
        - The actor: action distribution from the observation.
        - The critic: value estimate of the current state.

        Args:
            obs (torch.Tensor): Observation input.
        """
        # Calculate the value
        value = self.critic(obs)
        # Calculate actions
        actor_out  = self.actor(obs)

        if self.continuous:
            # Get σ from the log(σ), clamp it for stability.
            std = torch.exp(self.log_std).clamp(1e-6, 2.0)

            # The output of the actor network are numbers which represent the means (μ) of the Normal distribution. 
            # The final action a is a number sampled from that distribution.
            # D.Normal(actor_out, std) creates a normal distribution for each element such that
            # the mean is actor_out and σ is std.
            # D.Independent is a PyTorch wrapper that basically makes the distribution output a single number 
            # when using action_dist.log_prob(action). This is needed for the ratio in the loss:
            # ratio = π_θ(a | s) / π_θ_old(a | s). a is a vector; each element is independent, corresponding
            # to π_θ(a | s) = π_θ(a_1 | s) * π_θ(a_2 | s) * ... * π_θ(a_n | s). Note that the prior here is implied.
            # π_θ is action_dist.
            # In the log case, it will be Σ log(π_θ(a_i | s)).

            action_dist = D.Independent(D.Normal(actor_out, std), 1) # 1 corresponds here to sum over dim=-1, the action dim.
        else:

            # In the discrete case, the output needs to be converted to logits.
            # Thus, log softmax is used. Log softmax is preferred because it is more numerically stable
            # (avoiding overflow when logits are large in magnitude) than softmax.
            # softmax(x_i)      = exp(x_i) / Σ_j exp(x_j)
            # log_softmax(x_i)  = x_i - log(Σ_j exp(x_j))
            # Categorical(logits=...) internally uses log_softmax, so we do not need
            # to apply softmax ourselves. Also, as mentioned before, the ratio in the loss only
            # needs the log probabilities.
            action_dist = D.Categorical(logits=actor_out)

        return action_dist, value


class ClipSurrogatedObjectiveLoss(nn.Module):

    """
    The Clipped Surrogate Objective Loss is the main workhorse of PPO; it ensures that the current policy
    does not deviate too much from the old policy.

    [The probability ratio, r_θ]

    The probability ratio compares the probability of taking a certain action under the new policy versus
    the old policy:

        r_θ = π_θ_new(a | s) / π_θ_old(a | s)

    ---

    First, clip the probability ratio to make the policy updates more numerically stable:

        r_θ_clipped = clip(r_θ, 1 - ε, 1 + ε)

    Next, calculate the surrogate objective loss:

        L_CLIP = 𝔼[r_θ_clipped * (-A)] = -mean(r_θ_clipped * A)


    Where: 
    A : The advantage function. It is multiplied because we want to encourage
        increased expected accumulated reward (A > 0). The negative sign is included
        because this is a loss function that is minimized.

    ε : The clipping threshold.

    Args:

        eps (float): The clipping threshold , ε.
    """
    
    def __init__(
                self, 
                eps:float,
                ):
        super().__init__()

        self.eps = eps
        
    def forward(self, ratio, adv):
        """
        Calculate the Clipped Surrogate Objective Loss.

        Args:

            ratio (torch.tensor): The ratio tensor between the old policy and the new one. 
                                Shape: (num_envs * num_steps//num_minibatches,)
                                ratio = π_θ(a | s) / π_θ_old(a | s)
            
            adv (torch.tensor): The advantage tensor. Shape: (num_envs * num_steps//num_minibatches,)

            Note: num_steps//num_minibatches = number of steps per rollout.
        """

        return -1.0*torch.min(
            ratio*adv,
            torch.clamp(ratio,1-self.eps,1+self.eps)*adv
        ).mean()

class ValueFunctionLoss(nn.Module):
    """
    The Value Function Loss in PPO is the loss term regarding the critic; it ensures that the critic
    is performing its job correctly evaluating the value of the current state. It is defined as the mean squared error
    between the calculated value, V, and the accumulated discounted rewards, Gt which represent
    the true value of the state. 

    L_VALUE = -0.5 *  mean((V - Gt)^2) * coeff

    Args:
    coeff (float): The multiplier for the value loss in the PPO loss.
    """
    def __init__(
                self, 
                coeff:float,
                ):
        super().__init__()
        self.coeff = coeff

    def forward(self,Gt,V):

        """
        Calculate the Value Function Loss.
        
        Args:
            Gt (torch.tensor): The accumlated discounted reward tensor. 
                            Shape: (num_envs * num_steps//num_minibatches,)
            V  (torch.tensor): The value function output from the critic.
                            Shape: (num_envs * num_steps//num_minibatches,)

            Note: num_steps//num_minibatches = number of steps per rollout.
        """

        return self.coeff * 0.5 * torch.pow(V - Gt,2).mean()

class EntropyBonus(nn.Module):

    """
    The Entropy Bonus in PPO encourages exploration by increasing the randomness of the action distribution. 
    Higher entropy means the policy is less certain and explores more.
    This term is added to the loss function, thus multiplied by a negative sign,
    higher entropy reduces the loss, thereby encouraging exploration

    H = -Σp(x)ln(p(x)) (source : https://en.wikipedia.org/wiki/Entropy_(information_theory)#Definition)

    From torch's source code:
    entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(action_dist.scale)
    loss = -self.coeff* entropy

    Args:
        coeff (float): Multiplier for the entropy bonus term in the PPO loss.
    """

    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff

    def forward(self,action_dist):
        """
        Calculate the Entropy Bonus.

        Args:
        action_dist (torch.distributions.Distribution): The action distribution output from the actor.
        """

        return -self.coeff * action_dist.entropy().mean()

        
class PPOLoss(nn.Module):

    """
    The loss for PPO, composed of Clip Surrogated Objective Loss, Value Function Loss,
    Entropy Bonus, and a KL divergence regularizer.

    The total loss is as follows (light on notation on purpose):
    
        PPOLoss = 
            1.0       * L_CLIP(probability ratio, advantage)+
            value_c   * L_VALUE(value, accumulated discounted reward) +
            entropy_c * Entropy_Bonus(action_dist) +
            kl_coeff  * KL_DIV(probability ratio)
    
    KL_DIV is Kullback–Leibler divergence (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) and
    essentially is another regularizing term for the loss. It works similarly to the Clip Surrogated Objective Loss
    by punishing big changes in the policy.

    Args:
        eps (float): Clip Surrogated Objective Loss's clipping threshold , ε, 
        value_c (float): Coefficient for the value function loss term.
        entropy_c (float): Coefficient for the entropy bonus term.
        kl_coeff (float): Coefficient for the KL regularizer term.
    """

    def __init__(self, eps, value_c,entropy_c, kl_coeff, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clip_loss = ClipSurrogatedObjectiveLoss(eps)
        self.value_loss = ValueFunctionLoss(value_c)
        self.ent_bonus = EntropyBonus(entropy_c)

        self.kl_coeff = kl_coeff
        
    
    def forward(self ,adv , Gt, V, actions, actions_dist , old_logprob):
        """
        Compute the PPO loss.

        Args:
            adv (torch.Tensor): The advantages.
                Shape: (num_envs * (num_steps//num_minibatches),)
            Gt (torch.Tensor): The accumlated discounted reward tensor.
                Shape: (num_envs * (num_steps//num_minibatches),)
            V (torch.Tensor): The value function output from the critic.
                Shape: (num_envs * (num_steps//num_minibatches),)
            actions (torch.Tensor): Actions taken by the agent.
                Shape: (num_envs * (num_steps//num_minibatches), action_dim)  (continuous)
                       (num_envs * (num_steps//num_minibatches),)             (discrete)

            actions_dist (torch.distributions.Distribution): The action distribution output from the actor.
                Batch shape: (num_envs * (num_steps//num_minibatches),)
                Event shape: (action_dim,) (continuous) or () (discrete)

            old_logprob (torch.Tensor): Log-probabilities of the sampled actions under the old policy.
                Shape: (num_envs * (num_steps//num_minibatches),)

        Note: num_steps//num_minibatches = number of steps per rollout.
        """
        new_log_prob = actions_dist.log_prob(actions)
        log_ratio = new_log_prob - old_logprob
        ratio = torch.exp(log_ratio)

        # In general, KL = Σ P(x) * log(P(x)/Q(x)) and is meant to measure how different two probability distributions are.
        # In this case, we would like to measure how the new policy differs from the old one, and penalize it.
        #
        # Recall that π(a) is a probability distribution, so for a fixed state s: π = π(a | s).
        #
        # Thus, KL(π_old | π_new) = Σ π_old(a) * log(π_old(a) / π_new(a)) = Σ π_old(a) * -log_ratio(a)
        #
        # 𝔼[g(x)] = Σ g(x) * p(x) (https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician)
        #
        # and in this case p(x) = π_old(a), g = g(x) = log(π_old(a)/π_new(a)) = -log_ratio(a); thus this simplifies to
        #
        # KL = 𝔼[-log_ratio] = -𝔼[log_ratio]
        #
        # But using -mean(log_ratio) as an estimate can be numerically noisy on minibatches and can even become negative,
        # which is not ideal for something used as a loss. Thus we can do this trick (taken from cleanRL):
        #
        # Note that 𝔼[exp(log_ratio)] = Σ exp(log(π_old(a)/π_new(a)) * π_old(a)) = Σ (π_old(a)/π_new(a)) * π_old(a) =
        # = Σ π_new(a) = 1 
        # (π_new(a) is a probability distribution of actions, and we are summing along them; thus by definition it is 1).
        #
        # Next, we know that exp(x) ≥ 1 + x → exp(x) - 1 - x ≥ 0 → [x = log_ratio] → exp(log_ratio) - 1 - log_ratio >= 0.
        #
        # 𝔼[exp(log_ratio) - 1 - log_ratio] = 1 - 1 - 𝔼[log_ratio] = -𝔼[log_ratio].
        #
        # Putting it all together, we have an expression KL = exp(log_ratio) - 1 - log_ratio which is equivalent
        # to the original term, but is also ≥ 0. The mean is to approximate the expected value.

        kl = ((log_ratio.exp() - 1) - log_ratio).mean()

        clip_loss = self.clip_loss(ratio,adv)
        value_loss = self.value_loss(Gt,V)
        ent_bonus = self.ent_bonus(actions_dist)
        loss = clip_loss + value_loss + ent_bonus + self.kl_coeff * kl

        return  loss, kl

class GeneralizedAdvantageEstimation(nn.Module):
    """
    Generalized Advantage Estimation (GAE) , is a way to estimate the advantage 
    (how much is an action better for the given state) given the rewards and values.
    https://arxiv.org/pdf/1506.02438#page=5 , eq. 16

    At ≈  Σ(γλ)^l *δ_{t+l} 
    Where l = 0 -> num_steps

    δ_t is the Temporal Difference (TD) error. It means how much the reward 
    plus the next value differs from the current value estimate.

    δ_t = r_t + γ V(s_{t+1}) − V(s_t)
    
    Explanation:
    The Q value,
    Q(s,a) = 𝔼[G | s, a]
           = 𝔼[Σ(γ^k * r_{t+k} | s, a)]
           = 𝔼[r_t + Σ(γ^{k+1} * r_{t+k+1} | s, a)]
           = 𝔼[r_t | s, a] + γ * 𝔼[Σ γ^k * r_{t+k+1} | s, a]
           = 𝔼[r_t | s, a] + γ * 𝔼[G_{t+1} | s, a] 
           = 𝔼[r_t | s, a] + γ * 𝔼[V(s_{t+1}) | s, a] (*)
           = 𝔼[r_t + γ * V(s_{t+1}) | s, a] (**)
    
    (*) Why 𝔼[G_t | s ,a] = 𝔼[V(s_t) | s,a] ? 
    See    https://github.com/BenBenyamin/VerbosePPO/blob/main/images/1.png
    (**) Simillarly, V(s) = 𝔼[G | s] = 𝔼[r_t + γ * V(s_{t+1}) | s] 
    (https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf#page=158 , eq. 6.3-4)

    Thus, 
    A(s,a) = Q(s,a) - V(s) 
           = 𝔼[r_t + γ * V(s_{t+1}) | s, a] - V(s_t)
           = 𝔼[r_t + γ * V(s_{t+1}) - V(s_t) | s, a]
           = 𝔼[δ_t | s, a]
           This means that the A(s,a) is the expected value of the TD, δ_t given
           action a and state s. (Also eq. 10 in the original paper, https://arxiv.org/pdf/1506.02438#page=4)
    
    Define: Â_{t,k} := Σγ^l *δ_{t+l} 
            Where l = 0 -> k -1
            For example:
            Â_{t,0} = δ_t
            Â_{t,1} = δ_t + γ*δ_{t+1}
            Â_{t,2} = δ_t + γ*δ_{t+1} + γ^2*δ_{t+2}
            and in general: 
            Â_{t,n+1} = Â_{t,n} + γ^n*δ_{t+n}

    Note that:
            Â_{t,∞} = Σ γ^k *r_{t+l} - V(s) (Eq 15 : https://arxiv.org/pdf/1506.02438#page=4)
                    = G_t - V(s)
            Thus 𝔼[Â_{t,∞}] = 𝔼[G_t | s,a] - V(s) = Q(s,a) - V(s) = A(s,a)

    Therfore GAE is defined as
    GAE := Σ(γλ)^l *δ_{t+l}
    Where l = 0 -> ∞
    In practice: l = 0 -> num_steps
    Where γ ∈ [0,1] is the discount factor for future rewards. 
    γ = 0 : prioritize immediate rewards; γ = 1 : Treat future rewards equally.

    and   λ ∈ [0,1] is the decay parameter / GAE λ / eligibility trace parameter.
    small λ : advantages are short-sighted → high bias , low variance. 
    It is biased because there is a systematic error, due to "short-sightedness".
    Variance is low because less step rewards are used.
    
    large λ : advantages include many rewards far in the future → noisy → high variance low bias.
    Simillarly, has low bias because it includes many future rewards and better captures long-term effects. 
    Variance is high because summing many stochastic rewards makes the estimate noisy. 

    Note: 
    The time complexity for the forward-sum computation is Θ(num_steps^2). 
    Equivalently, using backward recursion:
    Â_t = δ_t + γ * λ * Â_{t+1}
    This runs in Θ(num_steps), but requires initializing
    Â_{num_steps} := 0 as the base case for the recursion. 
    An extra value V_{num_steps} is also needed to compute δ at the last timestep.

    Args:
        gamma (float, optional): γ , the discount factor for future rewards. Defaults to 0.99.
        lam (float, optional): λ, the GAE decay parameter. Defaults to 0.95.

    """
    def __init__(self, gamma=0.99, lam=0.95):
        super().__init__()
        self.gamma = gamma
        self.lam = lam

    def forward(self, rewards, values, dones, norm=True):
        """
        Compute Generalized Advantage Estimation (GAE).


        Args:
            rewards (torch.Tensor): Tensor of rewards at each timestep.
                Shape: (num_steps, num_envs)
            values (torch.Tensor): Tensor of value estimates for each state, including bootstrap for last state.
                Shape: (num_steps + 1, num_envs)
            dones (torch.Tensor): Tensor indicating episode termination (1 if done, 0 otherwise).
                Shape: (num_steps, num_envs)
            norm (bool, optional): If True, normalize the advantages to have mean 0 and std 1. Default: True.

        Returns:
            torch.Tensor: Tensor of advantage estimates.
                Shape: (num_steps, num_envs)
            torch.Tensor: Tensor of returns estimates.
                Shape: (num_steps, num_envs)
        """

        # Remove trailing singleton dimension from critic value outputs if needed
        if values.dim() == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)

        # Check if it is a single / vector environment
        if len(rewards.shape) == 1:
            num_steps = rewards.shape[0]
            num_envs = 1
        else:
            num_steps, num_envs = rewards.shape

        advantages = torch.zeros((num_steps, num_envs), device=rewards.device, dtype=rewards.dtype)
        last_gae = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)

        # Do the recursive backward sum
        for t in reversed(range(num_steps)):
            # If the episode ended at step t, values[t+1] belongs to the next episode
            #  and is therefore not meaningful.
            next_nonterminal = 1.0 - dones[t].float()
            # δ_t = r_t + γ V(s_{t+1})*next_nonterminal - V(s_t)
            # next_nonterminal is there to zero out values not from the same episode.
            delta = rewards[t] + self.gamma * values[t + 1] * next_nonterminal - values[t]
            # Â_t = δ_t + γ * λ * Â_{t+1}*next_nonterminal
            advantages[t]  = delta + self.gamma * self.lam * next_nonterminal * last_gae
            last_gae = advantages[t]
    
        # Q(s,a) = 𝔼[G_t | s,a], the expected return from taking action a in state s.
        # Since we cannot compute the expectation over all possible next states and rewards,
        # we approximate Q(s,a) with a sampled estimate of G_t.
        # By definition, A(s,a) = Q(s,a) - V(s), so equivalently:
        # G_t ≈ Q(s,a) ≈ A(s,a) + V(s)
        returns = advantages + values[:-1] # [:-1] because we need to un-bootstrap

        # Rewards can be very sparse, delayed, or have spikes.
        # Normalizing advantages prevents unusually large or small advantage values from 
        # destabilizing the policy updates, making learning smoother and more reliable.
        if norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        return advantages, returns

class PPOTrainer:
    def __init__(
        self,
        policy: ActorCritic,
        env,
        num_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_minibatches: int = 4,
        update_epochs: int = 4,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        kl_coeff: float = 0.02,
        max_grad_norm: float = 0.5,
        target_kl: float = None,
        learning_rate: float = 2.5e-4,
        anneal_lr: bool = True,
    ):
        """
        Handles the full PPO training loop: collects rollouts 
        from the environment, computes Generalized Advantage Estimates (GAE), 
        and updates the policy and value networks using minibatch stochastic
        gradient descent with the PPO clipped objective.

        Supports single and vectorized environments, continuous and discrete action spaces.
        Default values taken from CleanRL.

        Args:
            policy (ActorCritic): The actor-critic model containing the policy and value networks.
            env (gymnasium.Env | stable_baselines3.common.vec_env.VecEnv): The environment instance to train on (single or vectorized).
            num_steps (int, optional): Number of steps to collect per update. Default: 2048.
            gamma (float, optional): Discount factor for future rewards. Default: 0.99.
            gae_lambda (float, optional): Lambda parameter for Generalized Advantage Estimation (GAE). Default: 0.95.
            num_minibatches (int, optional): Number of minibatches to split the rollout into for each update. Default: 4.
            update_epochs (int, optional): Number of epochs to update the policy per rollout. Default: 4.
            norm_adv (bool, optional): Whether to normalize the advantages to mean 0 and std 1. Default: True.
            clip_coef (float, optional): PPO clipping coefficient for policy updates. Default: 0.2.
            vf_coef (float, optional): Coefficient for value function loss. Default: 0.5.
            ent_coef (float, optional): Coefficient for entropy bonus to encourage exploration. Default: 0.01.
            kl_coeff (float, optional): Coefficient for KL divergence penalty (if used). Default: 0.02.
            max_grad_norm (float, optional): Maximum gradient norm for clipping. Default: 0.5.
            target_kl (float, optional): Target KL divergence threshold for early stopping of policy updates. 
                                        If None, no early stopping based on KL is applied. Default: None.
            learning_rate (float, optional): Learning rate for the optimizer. Default: 2.5e-4.
            anneal_lr (bool, optional): Whether to linearly anneal the learning rate during training. Default: True.
        """

        
        self.policy = policy
        self.env = env
        # Check if vectorized or not
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vec = hasattr(env, "num_envs")
        self.obs_space = env.observation_space
        self.action_space = env.action_space

        self.device = next(policy.parameters()).device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_steps = num_steps

        # GAE module
        self.gae = GeneralizedAdvantageEstimation(gamma, gae_lambda)

        # PPO loss
        self.loss_func = PPOLoss(
            eps=clip_coef,
            value_c=vf_coef,
            entropy_c=ent_coef,
            kl_coeff=kl_coeff,
        )

        # Other hyperparameters
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr

        self.num_timesteps = 0 # How many timesteps in total where processed

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.learning_rate,
            ## Weight decay is just L2 regularization, 
            # which can hurt exploration for R, thus is set to 0.
            weight_decay=0.0,
            
        )

        self.anneal_lr = anneal_lr
            
    def rollout(self):
        """
        Rollout for on-policy PPO training.
        Runs the current policy in the environment for `num_steps` and records the transition data
        needed by `update()` (observations, actions, rewards, done flags, value estimates, and old log-probs).

        Returns:
            rewards (torch.Tensor): Tensor of rewards at each timestep.
                Shape: (num_steps, num_envs)
            values (torch.Tensor): Tensor of value estimates for each state, including bootstrap for last state.
                Shape: (num_steps + 1, num_envs)
            dones (torch.Tensor): Tensor indicating episode termination (1 if done, 0 otherwise).
                Shape: (num_steps, num_envs)
            obs_tensor (torch.Tensor): Tensor of observations at each timestep.
                Shape: (num_steps, num_envs)
            actions_tensor (torch.Tensor): Tensor of action taken by the policy at each timestep.
                Shape: (num_steps, num_envs)
            old_logprob (torch.Tensor): Log-probabilities of the sampled actions under the old policy.
                Shape: (num_steps,num_envs)
        """

        obs = self.env.reset()
        if not self.is_vec:
            obs = obs[0]

        rewards , values, dones , actions_list , obs_list , old_logprob = [], [], [] , [] , [] , []

        for _ in range(self.num_steps):
            
            obs_t = torch.as_tensor(obs, dtype=torch.float32 , device= self.device)
            obs_list.append(obs_t)

            actions_dist, value  = self.policy(obs_t)
            actions = actions_dist.sample()

            if self.is_vec:
                # SB3 VecEnv: step() → obs, rewards, dones, infos
                # Environment runs on the CPU, thus transfer is needed
                obs , reward, done_vec, infos = self.env.step(actions.cpu().numpy())

                # Extract truncated from info
                truncated_vec = torch.as_tensor(
                    [info.get("TimeLimit.truncated", False) for info in infos],
                    dtype=torch.bool,
                    device=self.device,
                )
                terminated_vec = torch.as_tensor(done_vec, dtype=torch.bool, device=self.device)
                done_flags = terminated_vec | truncated_vec
                
            else:

                obs, reward, terminated, truncated , _ = self.env.step(actions.cpu().numpy())
                done_flags = terminated or truncated

                if done_flags:
                    obs , _ = self.env.reset()
            
            done_flags = torch.tensor(done_flags, dtype= torch.bool, device = self.device)
            
            rewards.append(torch.as_tensor(reward, dtype=torch.float32 , device=self.device))
            values.append(value.detach())
            dones.append(done_flags)
            actions_list.append(actions.detach())
            old_logprob.append(actions_dist.log_prob(actions).detach())

            self.num_timesteps +=self.num_envs
   
        
        # Add value of the last observation (s_T)
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32 , device=self.device)
            _, last_value = self.policy(obs_t)
        values.append(last_value.detach())

        # Stack lists into batch tensors
        rewards = torch.stack(rewards).to(self.device)
        values = torch.stack(values).to(self.device)
        dones = torch.stack(dones).to(torch.bool).to(self.device)
        obs_tensor = torch.stack(obs_list).to(self.device)
        actions_tensor = torch.stack(actions_list).to(self.device)
        old_logprob = torch.stack(old_logprob).to(self.device)

        return rewards , values, dones, obs_tensor ,actions_tensor , old_logprob
                
    def update(self , rewards , values, dones, obs_tensor ,actions_tensor , old_logprob):
        """
        Updates the policy and critic using the on-policy data collected during the rollout.
        Shuffles the rollout batch and splits it into minibatches for training.

        Args:
            rewards (torch.Tensor): Tensor of rewards at each timestep.
                Shape: (num_steps, num_envs)
            values (torch.Tensor): Tensor of value estimates for each state, including bootstrap for last state.
                Shape: (num_steps + 1, num_envs)
            dones (torch.Tensor): Tensor indicating episode termination (1 if done, 0 otherwise).
                Shape: (num_steps, num_envs)
            obs_tensor (torch.Tensor): Tensor of observations at each timestep.
                Shape: (num_steps, num_envs)
            actions_tensor (torch.Tensor): Tensor of action taken by the policy at each timestep.
                Shape: (num_steps, num_envs)
            old_logprob (torch.Tensor): Log-probabilities of the sampled actions under the old policy.
                Shape: (num_steps,num_envs)
        """
        # Get the advantages and rewards from GAE
        advantages, returns = self.gae(rewards,values,dones,norm = self.norm_adv)

        # Get the random indecies for the minibatch
        batch_size = self.num_steps * self.num_envs
        minibatch_size = batch_size // self.num_minibatches
        indices = torch.randperm(batch_size, device=self.device)

        # flatten once per update
        obs_flat = obs_tensor.reshape(batch_size, -1)

        # Flatten the action tensors 
        if self.policy.continuous:
            actions_flat = actions_tensor.reshape(batch_size, -1) # new shape: (num_steps*num_envs,action_dim)
        else:
            actions_flat = actions_tensor.reshape(batch_size) # new shape: (num_steps*num_envs,)
    
        old_logprob_flat = old_logprob.reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)
        returns_flat = returns.reshape(batch_size, -1) # Match critic's output shape, (batch_size,1)

        for minibatch in range(self.num_minibatches):

            start = minibatch * minibatch_size
            end = min(start + minibatch_size, batch_size) ## avoid integer div edge case
            # Get minibatch
            mb_inds = indices[start:end]

            # select minibatch 
            mb_obs = obs_flat[mb_inds].to(self.device)
            mb_actions = actions_flat[mb_inds].to(self.device)
            mb_old_logprob = old_logprob_flat[mb_inds].to(self.device)
            mb_advantages = advantages_flat[mb_inds].to(self.device)
            mb_returns = returns_flat[mb_inds].to(self.device)
            
            new_actions_dist , new_values = self.policy(mb_obs)
            
            # Feed it into the loss
            loss , kl = self.loss_func(
                adv=mb_advantages,
                Gt=mb_returns,
                V=new_values,
                actions=mb_actions,
                actions_dist=new_actions_dist,
                old_logprob = mb_old_logprob,
            )

            # Check KL early stopping (if defined)
            if self.target_kl and kl > self.target_kl:
                break
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            # keep the gradient norm less than self.max_grad_norm
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

    
    def train(self, num_timesteps:int):
        """
        The training logic for PPO. First, collects an [on-policy] rollout (everything needed to update the policy),
        and then updates the policy for multiple epochs over that fixed batch. Also handles linear learning rate
        annealing/decay over the full training run.

        Notes:
            The number of time steps is treated as a global timestep budget. 
            Vectorized environments will share this budget with each other. 

        Args:
            num_timesteps (int): The aggregate number of environment timesteps to train for.
        """

        while self.num_timesteps < num_timesteps:
            
            # Linear annealing
            if self.anneal_lr:
                frac = 1 - (self.num_timesteps / num_timesteps)
                lr_now = self.learning_rate * frac
                for g in self.optimizer.param_groups:
                    g["lr"] = lr_now

            # Collects on policy rollout
            rewards , values, dones, obs_tensor ,actions_tensor , old_logprob = self.rollout()

            # Update for how many epochs defined
            for _ in range(self.update_epochs):

                self.update(rewards , values, dones, obs_tensor ,actions_tensor , old_logprob)

class PPO:
    """
    A Wrapper class to encompass the PPO pipeline: Defining the policy, training, perdicting ,.. 
    Wrapper class that wires together the PPO pipeline.
    Builds an Actor-Critic policy network from the environment, handles rollouts, GAE, and PPO updates. 
    Exposes a single entry point for training and inference.
    
    Args:
            env (gymnasium.Env | stable_baselines3.common.vec_env.VecEnv): The environment instance to train on (single or vectorized).
            device: The device on which training will run. Default: "cpu".
            actor (nn.Module, optional): Optional custom actor network. If None then the one from cleanRL is used.
                                         Default : None.
            hidden_size (int): The size of the hidden layer for the actor critic network. Default: 64.
            num_steps (int, optional): Number of steps to collect per update. Default: 2048.
            gamma (float, optional): Discount factor for future rewards. Default: 0.99.
            gae_lambda (float, optional): Lambda parameter for Generalized Advantage Estimation (GAE). Default: 0.95.
            num_minibatches (int, optional): Number of minibatches to split the rollout into for each update. Default: 4.
            update_epochs (int, optional): Number of epochs to update the policy per rollout. Default: 4.
            norm_adv (bool, optional): Whether to normalize the advantages to mean 0 and std 1. Default: True.
            clip_coef (float, optional): PPO clipping coefficient for policy updates. Default: 0.2.
            vf_coef (float, optional): Coefficient for value function loss. Default: 0.5.
            ent_coef (float, optional): Coefficient for entropy bonus to encourage exploration. Default: 0.01.
            kl_coeff (float, optional): Coefficient for KL divergence penalty (if used). Default: 0.02.
            max_grad_norm (float, optional): Maximum gradient norm for clipping. Default: 0.5.
            target_kl (float, optional): Target KL divergence threshold for early stopping of policy updates. 
                                        If None, no early stopping based on KL is applied. Default: None.
            learning_rate (float, optional): Learning rate for the optimizer. Default: 2.5e-4.
            anneal_lr (bool, optional): Whether to linearly anneal the learning rate during training. Default: True.
            actor_std (float, optional): Standard deviation for continuous actor distribution. Defaults to 0.01.
            init_weights(bool, optional): Whether to weight initialization or not. Default: True.
            critic_std (float, optional): Initialization scale for critic output layers. Defaults to 1.0.
            hidden_std (float, optional): Initialization scale for hidden layers. Defaults to √2.
            bias_const (float, optional): Constant value to initialize network biases with. Defaults to 0.0.

    """
    def __init__(
        self,
        env,
        device = "cpu",
        actor:nn.Module = None,
        hidden_size=64,
        num_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=4,
        update_epochs=4,
        norm_adv=True,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        kl_coeff=0.02,
        max_grad_norm=0.5,
        target_kl=None,
        learning_rate=2.5e-4,
        anneal_lr=True,
        actor_std:float=0.01,
        init_weights:bool= True,
        critic_std:float=1.0,
        hidden_std:float = np.sqrt(2),
        bias_const:float=0.0,
    ):
        
        self.env = env

        obs_dim  = env.observation_space.shape[0]

        self.device = device

        if isinstance(env.action_space, gym.spaces.Discrete):
            continuous = False
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            continuous = True
            action_dim = env.action_space.shape[0]
        else:
            raise ValueError("Unsupported action space type:", env.action_space)

        action_dim = int(action_dim)

        self.policy = ActorCritic(
            actor=actor,
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            continuous=continuous,
            init_weights=init_weights,
            actor_std=actor_std,
            bias_const=bias_const,
            critic_std=critic_std,
            hidden_std=hidden_std,

            ).to(self.device)

        self.trainer = PPOTrainer(
            policy=self.policy,
            env=self.env,
            num_steps=num_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            num_minibatches=num_minibatches,
            update_epochs=update_epochs,
            norm_adv=norm_adv,
            clip_coef=clip_coef,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            kl_coeff=kl_coeff,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            learning_rate=learning_rate,
            anneal_lr=anneal_lr,
        )

    def learn(self, total_timesteps):
        """
        Trains the PPO agent for a given number of environment timesteps by delegating to PPOTrainer.train().

        Args:
            total_timesteps (int): Total number of environment timesteps to train for.

        Returns self to allow chaining (e.g., model.learn(...).save(...)).
        """
        self.trainer.train(total_timesteps)
        return self

    def predict(self, obs, deterministic=True):
        """
        Computes an action from the current policy given an observation.

        For continuous action spaces, deterministic mode returns the distribution mean.
        For discrete action spaces, deterministic mode returns the argmax action.
        If deterministic is False, actions are sampled from the policy distribution.

        Args:
            obs (torch.Tensor | np.array): Observation(s) to act on.
                Shape: (batch_size, obs_dim).
            deterministic (bool): If True, use mean/argmax action. If False, sample from the policy.

        Returns:
            np.ndarray: Action(s) as a numpy array on the CPU.
                Shape:
                    Discrete: (batch,)
                    Continuous: (batch_size, action_dim)
        """
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device)

        with torch.inference_mode():
            dist, _ = self.policy(obs)

            if self.policy.continuous:
                if deterministic:
                    action = dist.mean
                else:
                    action = dist.sample()
            else:
                if deterministic:
                    action = torch.argmax(dist.probs)
                else:
                    action = dist.sample()

        return action.cpu().numpy()

    def save(self, path):
        """
        Saves the policy to disk.

        Args:
            path (str): File path to save the policy weights to.
        """
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        """
        Loads policy parameters from disk into the current policy.

        Args:
            path (str): File path to load the policy weights from.
        """
        self.policy.load_state_dict(torch.load(path))
