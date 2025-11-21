### **Meeting Notes**

**Topic:** Hackathon Team Kick-off and Project Planning
**Attendees:**
*   **Darshan:** Scientific Researcher at Lazzey, background in Computational Science.
*   **Szymon Mazurek:** Deep Learning Engineer from Cyfronet (the facility hosting the cluster).
*   **Dragosh:** Student at the Romanian-American University and AI Research Center member.
*   **Sorin:** Final-year Computer Science student in Romania.
*   **Andrei:** Lecturer at the Romanian-American University and Head of the AI Center.
*   **Hackathon Organizer** (Joined at the end)

*(Note: A fourth team member, Titus, was mentioned but was not present).*

---

### **Meeting Summary & Key Discussion Points**

**1. Team Introductions & Roles:**
*   The meeting began with each member introducing themselves, their background, and their expertise.
*   Szymon from Cyfronet will serve as the primary contact for any issues related to the HPC cluster (Helios), including performance and troubleshooting.
*   Andrei is leading the project from a research perspective, with Darshan, Dragosh, and Sorin contributing as researchers and developers. Most of the student members have limited prior experience with HPC, which is a key reason for participating in the hackathon.

**2. Project Goal & Scientific Context:**
*   The primary goal of the project is to create a model that can **recreate raw EEG signals**.
*   The proposed architecture is a hybrid model combining **Graph Neural Networks (GNNs)** and **Spiking Neural Networks (SNNs)**.
*   The long-term vision, as explained by Andrei, is to see if this model can serve as a "digital representation of the cortex." By successfully recreating the signals, the team hopes to gain insights into how the brain generates them, potentially by analyzing the model's internal activations and learned structures.

**3. Technical Approach & Challenges:**
*   **Data:** The team is using a public EEG dataset with 31 subjects, 128 channels, and 10-minute recordings at a 1000Hz sampling rate. They are initially focusing on the "resting state" data.
*   **Data Pre-processing:** A significant portion of the discussion focused on the noisy nature of EEG data.
    *   The team plans to perform pre-processing to remove artifacts.
    *   A key challenge is finding the right balance: over-cleaning the data might make it easier to process but would prevent the model from learning to generate realistic, noisy signals. The plan is to start with basic artifact removal and iterate.
*   **Model Architecture:** The current prototype uses a Graph Attention (GAT) layer. Szymon suggested investigating **GATv2**, a newer version that reportedly fixes issues with static attention and could improve performance.
*   **Training & Evaluation:**
    *   The initial concept is a reconstruction task, similar to an autoencoder.
    *   The discussion evolved toward a more complex **prediction task**: using the first nine minutes of a recording to predict the final minute.
    *   A critical point was raised about data splitting: simply using random time windows for testing is insufficient. A more robust evaluation will involve holding out entire subjects from the training set to see if the model can generalize to new, unseen individuals.

**4. Logistics & Next Steps:**
*   **Code & Resources:** The project's initial codebase is already on a public GitHub repository. Dragosh has been working on getting it to run locally but ran into memory limitations, which necessitates the use of the HPC cluster.
*   **Cluster Usage:** The immediate next step is for all team members to get access to the Helios cluster. The team will then move the data and code to the cluster and perform a simple run to ensure the environment is set up correctly.
*   **Hackathon Support:** A hackathon organizer joined at the end to confirm that a training session on using the cluster's software environment and scheduler (Slurm) will be offered. The Doodle poll for scheduling this session has been shared in Slack.

