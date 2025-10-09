OpenAAC Hackathon Application


*Team Name
Biologically Artificial (open to suggestions)





*Tell us about your team

Our team is a dynamic collaboration of three undergraduate students and a university lecturer, brought together specifically for this hackathon to transform our shared ideas into a working prototype. This event presents an exciting opportunity to leverage GPU clusters and cutting-edge resources to build something impactful.
We bring a diverse and complementary set of skills to the table, spanning UI/UX design, neuroscience, machine learning, artificial intelligence, as well as strong proficiency in Python and C++. What truly sets us apart is our shared passion, creativity, and relentless drive to bring our vision to life.
Our team is guided by a university lecturer with a strong research background in ML, computational biomechanics, neural networks, and neuroscience. His academic insight and technical expertise ensure a scientifically grounded and rigorous approach to development, allowing us to iterate quickly from concept to prototype without compromising on quality or credibility.
Together, we are ready to take on this challenge and make the most of the hackathon environment to build something innovative and meaningful.


Andrei
Luchici
andrei.luchici@rau.ro
Romania
Romanian-American University
Dragos
Velicu
velicu.m.mihaildragos23@stud.rau.ro
Romania
Romanian-American University
Titas
Ramancauskas
r045245m@student.staffs.ac.uk
United Kingdom
University of Staffordshire
Sorin
Turculet
sorin.turculet@stud.ubbcluj.ro


Romania
Babeș-Bolyai University





*What is the name of your application?
- NeuroGraph Inverse Solver (NGIS) (open to suggestions)


*Application Details
(Please provide an abstract of your application/code that includes the aim or goal of the application, what architecture the application uses (i.e. CPUs, GPUs), and any relevant information (i.e. GitHub repo link).)


- NGIS is an advanced inverse modelling framework designed to reconstruct subject-specific functional brain networks directly from high-density EEG recordings (128 channels). Starting with raw time-series data, we train a graph-structured spiking neural network (G-SNN), where each node represents a cortical neuron modelled using leaky integrate-and-fire dynamics, and edges encode synaptic connectivity.
The model is trained to minimise the difference between simulated and real EEG signals while adhering to biologically plausible constraints. Once trained, NGIS is capable of generating realistic brain activity for previously unseen subjects, enabling quantitative validation and potential generalisation across individuals.
The codebase is primarily implemented in Python, with performance-critical components selectively delegated to C++. We leverage PyTorch and PyTorch Geometric for graph-based learning, Brian2 for efficient spiking neural network simulation, and custom CUDA kernels for optimised low-level operations. During the hackathon, we will focus on porting the most computationally intensive kernels to multi-GPU A100 nodes using NCCL and Distributed Data Parallel (DDP) for scalability.
A public GitHub repository will be made available by August 5, 2025, and the project will be released under the Apache 2.0 open-source license.


(repo placeholder: https://github.com/placeholder/ngis-opensrc)
Target hardware: x86_64 CPUs for pre‑processing + NVIDIA GPUs for training/inference. Primary target: 4 × A100 (80 GB) nodes interconnected with NVLink.














*Application Domains


Computer Science
Machine Learning and AI
Neuroscience
(We also have the “other” option , but I believe these cover the entire scope of the project - Dragos)






Application area of focus
AI


Programming language(s) used in your application
AI Frameworks
C++
Python


Libraries used in your application.
PyTorch, PyTorch Geometric
Brian2 / BindsNET
NumPy, pandas, scikit‑learn
MNE‑Python (EEG I/O)
NetworkX
CUDA, cuDNN, NCCL

*What framework(s) and models have you used to work with your data?
(Is your model similar to ResNet-50 CNN, LSTM, BERT, random forest, etc.? What optimisers and/or training methods are you using? What systems have you worked on before with your data and models?)
Graph Neural Networks with message‑passing (GAT / GraphSAGE)
Spiking Neural Networks (Leaky‑Integrate‑and‑Fire & Izhikevich neurons)
Diffusion‑style generative prior for regularising latent connectivity
Optimiser: AdamW with cyclical LR; gradient clipping at 1.0; mixed‑precision (FP16/BF16).
Previous systems: single RTX 3080 desktop and 1×A100 node on HPC systems.


*Algorithmic motifs.
(Describe what types of algorithms dominate your application, especially the ones your team is targeting for acceleration.)
Our application is dominated by a combination of graph-based deep learning and spiking neural network (SNN) simulation algorithms, both of which are computationally intensive and well-suited for GPU acceleration.
Graph Neural Network (GNN) Training: We use PyTorch Geometric to implement and train graph-structured models where each node corresponds to a neuron and edges represent synaptic links. This involves message passing, neighbourhood aggregation, and backpropagation across irregular graph topologies—all of which benefit significantly from GPU parallelism.


Spiking Neural Network (SNN) Dynamics: The neural activity is modelled using leaky integrate-and-fire (LIF) dynamics within Brian2. This simulation step requires solving many small, sparse differential equations in parallel, which can be offloaded to CUDA kernels for speed.


Inverse Modelling / Optimisation Loop: Training the G-SNN involves minimising the discrepancy between simulated and observed EEG signals. This iterative optimisation is computationally intensive, especially when enforcing biological constraints across time steps and network structure.


For the hackathon, we are targeting the acceleration of:
Custom CUDA kernels for LIF neuron updates and synaptic transmission


Multi-GPU parallelism using NCCL and PyTorch Distributed Data Parallel (DDP) to scale training across A100 nodes


By accelerating these key algorithmic components, we aim to dramatically reduce training time and enable more complex and biologically accurate models within the hackathon timeframe.




*What is your current application performance?
(Describe the current performance characteristics of your application. Where does it run (CPU, GPU)? How many nodes does it scale to?)
The application is currently under active development, and formal performance benchmarking has not yet been conducted. At this stage, the core components run on a single GPU (A100-class or equivalent) with preliminary support for CPU fallback.
While we have not yet scaled the application across multiple nodes, the design anticipates distributed execution. Specifically, we plan to use PyTorch Distributed Data Parallel (DDP) and NCCL to scale the training of our graph-structured spiking neural networks across multi-GPU and multi-node environments.
During the hackathon, one of our key goals is to identify and accelerate bottlenecks by:
Profiling CUDA-based SNN simulation and graph operations


Porting compute-heavy kernels to support multi-GPU execution


Establishing baseline performance metrics


This will lay the groundwork for efficient large-scale simulations and generalisation across subject datasets.




Provide details about your data source and model (For Al codes only).
(Details should include information about licenses, size of the data source, size of the model, current time to train, and links to any applicable information.)
Temple University Hospital EEG Corpus (~1.5 TB, CC BY‑NC)
BCI Competition IV‑2a (9 subjects, 128 channels, CC BY‑NC)
~30 h of new 128‑channel EEG to be acquired in‑house using a Brain Products actiChamp gel-based system


*List the computing facilities this application runs on.
(Example: Desktop, local clusters, HPC centers, etc.)
Developer desktops (RTX 3080)
Center for Research in AI (CRAI)’s computing resources 


Projects brought to the event are required to have a license attached and detailed in the application. A Permissive-style Open Source License (e.g. BSD, MIT, or Apache 2.0 license) is preferred to ensure that most mentors can collaborate on the application(s).
Apache 2.0
Please provide further details on your License(s) if needed. ( This is optional and we can leave it blank )


* What do you hope to achieve at the Hackathon?
Our primary goals for the hackathon are:
Deliver an open-source, reproducible pipeline: We aim to package the full NGIS workflow using Docker and/or Singularity containers to ensure portability, transparency, and reproducibility for both research and clinical communities.


Produce a short paper or poster draft: We intend to generate a concise, well-structured manuscript or poster abstract suitable for submission to a neuroinformatics conference or workshop, summarising our approach, findings, and early performance insights.


Lay the foundation for future research: This project serves as a proof of concept for a larger research initiative. A successful outcome would support applications for future grants and multi-institution collaborations in computational neuroscience and AI-driven healthcare.



Potential Real-World Applications
Our framework has a broad range of translational applications, including:
Brain Tumor Detection and Localization
 Simulate "lesioned" brain activity by silencing nodes or adding synthetic anomalies in the model. Comparing the resulting EEG to a patient’s real data could help flag regions for targeted imaging or surgical intervention.


Patient-Specific Seizure Focus Mapping
 Run the model forward under various scenarios to identify cortical regions likely responsible for seizure onset, guiding surgical resection or laser ablation with higher precision.


Rapid Brain-Computer Interface (BCI) Calibration
 Generate realistic, labelled synthetic EEG data to significantly reduce the time needed for BCI headset training—from hours to minutes, especially valuable in clinical or assistive technology settings.


Closed-Loop Neurostimulation Design
 Use the digital twin to prototype and test neurostimulation protocols (e.g., TMS, tDCS, DBS) safely and non-invasively, accelerating therapy development.


Drug Discovery and Neurotoxicity Screening
 Simulate network-level effects of experimental compounds to assess neurological impact before moving to animal studies, helping streamline preclinical pipelines.


Is there anything else you'd like to mention or ask us?
We would appreciate early access to the cluster documentation (node specs, interconnect, scheduler). Thank you!





