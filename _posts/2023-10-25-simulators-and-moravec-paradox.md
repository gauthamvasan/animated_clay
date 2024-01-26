---
layout:     post
title:      "Moravec's paradox, Sim-to-Real Transfer & Robot Learning"
date:       2023-10-25 15:10:00
excerpt:    ""
---

<div class="row">
    <div class="col-xs-10">
        <p class="pubd">
            <img src="/img/moravec_paradox.png">
        </p>
    </div>
</div>


## Moravec's paradox
Early Artificial Intelligence (AI) researchers focused on tasks they found challenging, like games and activities requiring reasoning and planning. Unfortunately, they often overlooked the learning abilities observed in animals and one-year-olds. The effectiveness of our deliberate reasoning process stems from a potent, though typically unconscious, foundation of sensorimotor knowledge. Our adeptness in perceptual and motor learning allows us to effortlessly perform tasks such as walking, running, and recognizing objects. These sensorimotor skills, honed over millions of years of evolution, contrast with the comparatively recent development of abstract thinking. Notably, navigating the complexities of robotics remains a formidable challenge.

There has been impressive headway in robotics research in recent years, largely driven by the strides made in machine learning. 
While the realm of AI research is currently heavily dominated by Large Language Model (LLM) researchers, there's still a notable upswing in the enthusiasm for robotics research. 
In fact, works like Google's [RT-2](https://www.deepmind.com/blog/rt-2-new-model-translates-vision-and-language-into-action) tantalizingly dangle the prospect of embodied AGI being just around the corner.
For folks unfamiliar with the term, Artificial General Intelligence (AGI) refers to a hypothetical type of intelligent agent that, if realized, could learn to accomplish any intellectual task that human beings or animals can perform. 

The integration of LLMs with robots is an exciting development mainly because it finally enables us to communicate with robots in a way that was once confined to the realm of science fiction. 
However, the current use of LLMs has been more focused on symbolic planning, requiring additional low-level controllers to handle the sensorimotor data. 
<i>Despite the captivating demonstrations, it's important to note that the foundational issues of Moravec's paradox still persist... </i>

## A Line of Attack for Moravec's Paradox?
In a recent [TED talk](https://youtu.be/LPGGIdxOmWI?si=Wq-C17pjX_LI2lS5), [Prof. Pulkit Agrawal](https://people.csail.mit.edu/pulkitag/) argues that while evolution required millions of years to endow us with locomotion priors/skills, but the development of logic & reasoning abilities occurred more swiftly, driven by the presence of underlying learning capabilities. 
His core argument is that simulation can compensate for millions of years of evolution, allowing us to acquire the necessary priors and [inductive biases](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21483/). 

Simulators can provide a potentially infinite source of data for training robotic systems, which can be cost-prohibitive or impractical to obtain in the real world. 
Additionally, using simulations alleviates safety concerns associated with training and testing on real robots. 
Hence, there's a LOT of interest in learning a control policy purely in simulation and deploying it on a robot. 
This line of research, popularly known as [sim-to-real](https://arxiv.org/abs/2009.13303v2) in robot learning, refers to the process of transferring a robotic control policy learned in a simulated environment to the real world.

The sim-to-real transfer process typically involves training a robotic control policy in a simulated environment and then adapting it to the real world. 
This adaptation is necessary due to the differences between simulation and reality, such as sensory noise, dynamics, and other environmental factors. 
Typically, there is a significant gap between simulated and real-world environments, which can lead to a degradation in the performance of policies when transferred to real robots. 

## Sim-to-Real: The Silver Bullet? 
The basic idea behind the generative AI revolution is simple: Train a big neural network with a HUGE dataset from the internet, and then use it to do various structured tasks. 
For example, LLMs can answer questions, write code, create poetry, and generate realistic art. 
Despite these capabilities, we're still waiting for robots from science fiction that can do everyday tasks like cleaning, folding laundry, and making breakfast. 

Unfortunately, the successful generative AI approach, which involves big models trained on internet data, doesn't seamlessly scale to robotics. 
Unlike text and images, the internet lacks abundant data for robotic interactions. 
Current state-of-the-art robot learning methods require data grounded in the robot's sensorimotor experience, which needs to be slowly and painstakingly collected by researchers in labs for particular tasks. 
The lack of extensive data prevents robots from performing real-world tasks beyond the lab, such as making breakfast. Impressive results usually stay confined to a single lab, a single robot, and often involve only a few hard-coded behaviors. 

> Drawing inspiration from Moravec's paradox, the success of generative AI, and [Prof. Rich Sutton](http://incompleteideas.net/)'s post on [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html), the robot learning community's biggest takeaway so far is that we do not have enough data.

While there is some validity to criticisms, like [The Better Lesson](https://rodneybrooks.com/a-better-lesson/) by Dr. Rodney Brooks, we can all still agree that we're going to need a lot of data for robot learning. 

<i> The real question is, where does that data come from? </i>
Currently, I see three data sources, and it's worth noting that they do not have to be independent of each other:
1. Large-scale impressive efforts, like [Open X-Embodiment](https://arxiv.org/abs/2310.08864), where a substantial group of researchers collaborated to collect robot learning data for public use.
2. Massive open world simulators which can be used for Sim-to-Real Transfer for Robot Learning.
3. [Real-Time Learning on Real World Robots](https://drive.google.com/file/d/1GxgFf2eIpgE-JIO3nzI-lPab9BkVnxll/view)! 
    - This may be painstakingly slow and tedious. But this is also a necessary feature if we envision truly intelligent machines that can learn and adapt on the fly as they interact with the world. 

For points 1 & 3, the downside is the amount of human involvement required. 
It is incredibly challenging to autonomously collect robot learning data, which can indeed pose a significant barrier. 
Now let's talk about why most roboticists are currently paying a lot of attention to sim-to-real learning methods.

### ✅ The Appeal of Sim-to-Real 
1. <b>Infinite data</b>: Simulators offer a potentially infinite source of training data for robots, as acquiring large amounts of real-world robot data is often prohibitively expensive or impractical.
2. <b>The pain of working with hardware</b>: Researchers often favor simulators in their work due to the convenience of avoiding hardware-related challenges. Simulated environments provide a controlled and reproducible setting, eliminating the need to contend with physical hardware issues, allowing researchers to focus more on algorithmic and learning aspects. 
    - I love this sentiment shared by [Prof. Patrick Pilarski](https://sites.ualberta.ca/~pilarski/), my M.Sc advisor: "Robots break your heart. They break down at the worst possible moment - right before a demo or a deadline." 
3. <b>Differentiable physics</b>: There are simulators with differentiable physics. This fits nicely with certain approaches, especially with trajectory optimization in classical robotics. This also helps with estimating gradients of the reward or gradient of a value function with respect to the state in RL.
    - *In the real world, we do not have differentiable physics.*
4. <b>Domain randomization</b> is a crucial step in all sim-to-real approaches. This has more to do with robustness to changes rather than online adaptation. With sim-to-real, we want to expose the agent to as many scenarios as possible in the simulator before deployment. OpenAI's [solving a Rubik's cube with a robot hand](https://openai.com/research/solving-rubiks-cube_) demo is a fantastic showcase of this approach. 
    -  *The focus is not really on learning on the fly, but rather being robust to perturbations.*
5. <b>World models</b>: The simulator is a pseudo-reinforcement-learning-model which can help learn a value function better (assuming its a good simulator)
    - *This, however is not the same as learning a world model, rather trying to replicate the world to solve a very specific real-world task*"


While sim-to-real has its merits, I believe it may not be sufficient as there are key limitations that still need to be addressed.

### ❌ Limitations of Sim-to-Real 
1. <b>Sim-to-Real Gap</b>: We are limited in our ability to replicate the real world. Our simulators, for example, cannot faithfully replicate friction or contact dynamics. Small errors in simulations can compound and significantly degrade the performance of policies learned in simulation when applied to real robots. 
2. <b>Accuracy of simulators</b>: While techniques such as [domain randomization](https://lilianweng.github.io/posts/2019-05-05-domain-randomization/), and domain adaptation can help mitigate these limitations, they may not be sufficient for tasks requiring detailed simulations, especially in domains like agricultural robotics. 
3. <b>Cost of building simulators</b>: Another concern for me is that no one discusses the computational expenses associated with simulators. High-fidelity simulators can be extremely costly to both develop and maintain. 

To answer the question posed at the start of this section - No, but... 
I believe we can benefit immensely by incorporating simulators into our learning methods when appropriate. The issue is more nuanced than a simple yes or no :)

## Closing Thoughts
In the research community, I've noticed the widespread adoption of the term 'Out-of-distribution' (OOD). In robot learning, it denotes the challenge of handling data that deviates from the training data. Personally, I loathe this term. By the very nature of the learning problem, we acknowledge that preparing robots for all conceivable scenarios in advance is impossible. If we could predict every scenario, the need for learning methods would be obsolete! OOD essentially makes a case for integrating the test set into the training set, a notion that was once considered blasphemous in machine learning before the era of LLMs. 

Our current robot learning playbook seems to involve throwing a vast amount of data at our limited, not-so-great algorithms in simulation via domain randomization, with the hope that it encompasses all potential deployment scenarios. In addition, there's also a huge aversion to learning directly on the robot for an extended period of time. In my opinion, this strategy is doomed to fail because we possibly cannot, and will not be able to, model the entire world with a high degree of precision and accuracy.

I do believe that pre-training models using sim-to-real can serve as a good starting point. It can be a fantastic litmus test to rule out ineffective approaches. But clearly, we cannot expose our robots to all possible scenarios via training in simulation. The general-purpose robots we dream of must also have the ability to learn on-the-fly as they interact with the physical world in <i>real-time</i>. When learning in real-time, the real world does not pause while the agent computes actions or makes learning updates. Moreover, the agent obtains sensorimotor information from various onboard devices and executes action commands at a specific frequency. Given these constraints, a real-time learning agent must compute an action within a chosen action cycle time and perform learning updates without disrupting the periodic execution of actions. I believe significant improvements in scaleable learning methods, tailored to meet real-time requirements, are essential to take a significant step towards achieving truly effective learning on real-world robots.

It's likely a mistake to believe there's only one correct approach, as this can lead to unnecessary and unproductive arguments. It's sufficient to justify a research direction by emphasizing its interest, importance, and, in some cases, essentiality. Adopting this perspective makes it clear that both sim-to-real and real-time learning are crucial aspects of the learning process. They complement each other rather than being alternatives. Moreover, it appears that for many scenarios, real-time learning becomes essential. Regardless of the agent's initialization, the ability to learn in real-time becomes a fundamental requirement. 

I believe we are making significant strides in refining sim-to-real methods. Nevertheless, unless coupled with real-time learning abilities, our vision of achieving embodied AGI may remain unfulfilled.

    
### Addendum
- I came across another great post in a similar vein: [Will Scaling Solve Robotics?: Perspectives From Corl 2023](https://nishanthjkumar.com/Will-Scaling-Solve-Robotics-Perspectives-from-CoRL-2023/). Definitely worth a read! 
- This article is pretty neat too: Kaelbling, L. P. (2020). [The foundation of efficient robot learning](https://dspace.mit.edu/bitstream/handle/1721.1/130244/psKaelbling_v1c.pdf?sequence=2&isAllowed=y). Science, 369(6506), 915-916.

### Acknowledgements
Thanks to [Prof. Rupam Mahmood](https://armahmood.github.io/) and [Shivam Garg](https://svmgrg.github.io/) for several thoughtful discussions on this topic that have greatly shaped my views.