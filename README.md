# AI Flight Simulation: Reinforcement and Imitation Learning Environment

This project is a simulation environment developed using Python, featuring custom-modeled flight mechanics and failure/termination states. The primary objective is to develop an autonomous Artificial Intelligence (AI) agent that learns to survive and maximize flight performance under simulated physical conditions.

The system integrates **Reinforcement Learning** for autonomous agent development and **Imitation Learning** for training via user data, bringing both algorithms together within a unified architecture.

## Key Features

* **Custom Physics and Environmental Dynamics:** Gravity interactions, flight mechanics, and simulation termination (failure) conditions were built entirely from scratch.
* **Reinforcement Learning (RL) Architecture:** The autonomous agent updates its policy using reward and penalty metrics obtained through environmental interactions (trial-and-error), aiming for a higher score with each iteration.
* **Imitation Learning:** The system analyzes flight data from a human operator. The agent utilizes this data to integrate expert behaviors into its own model, refining its decision-making mechanism.

## Simulation Modes

The system offers three primary modes for different testing and training scenarios:

* **Manual Mode (Free Flight):** The standard control mode where learning algorithms are disabled, allowing the user to freely test the flight dynamics and physics engine.
* **Autonomous Learning (AI Reinforcement):** The mode where the AI takes full control. The agent analyzes environmental states, conducts autonomous flight tests, and establishes an optimal flight policy by learning from its mistakes.
* **User-Guided Learning (Imitation/Teaching):** The user pilots the aircraft while the system continuously logs state-action pairs in the background, actively training the AI based on the user's performance.

## Dataset Management and Agile Methodology

The dataset creation and processing pipeline for our Imitation Learning model is designed not as a static structure, but by utilizing **Agile** principles and the **Scrum** methodology:

* **User Story-Driven Data Generation:** The agent's core data collection objective is built on the following user story: *"As an autonomous pilot, I want to analyze the responses of an expert human operator and model these movements to produce optimal flight performance (business value)."*
* **Flight Sessions as Sprint Cycles:** Each flight session conducted in "User-Guided Learning" mode is treated as a **Sprint** within our data pipeline. Every maneuver by the user and the instantaneous environmental conditions are added to the system's **Product Backlog** (dataset) as new training data.
* **Sprint Retrospective and Continuous Improvement (CI):** The process of updating model weights with the collected data functions as a **Sprint Retrospective**. The agent analyzes its past errors by referencing the expert data and adapts its policy. This process ensures the Continuous Improvement of the model through the principles of transparency, inspection, and adaptation.

## Installation and Execution Guidelines

You can follow the steps below to run the project in your local environment:

**Option 1: Compiled Version (Standalone Execution)**
To test the simulation directly without needing a development environment:
1. Navigate to the `pythonApplication2/dist` directory within the project folder.
2. Run the executable `.exe` file located in this directory.

**Option 2: Developer Environment (Via Source Code)**
To review the codebase and compile the project via an IDE:
1. Open the project using **Visual Studio** or your preferred Python IDE.
2. Ensure that the required Python dependencies are installed, and run the project from the root directory.

---
**Development Language:** Python
