"""
Unified Runner for Lab Sheet-08: Mini Projects on Deep Learning and Reinforcement Learning.
MCA III Semester (Session 2026-2027) - COER University
"""

import os
import sys
import time

def run_project1():
    print("\n" + "#" * 70)
    print("# PROJECT 1: HANDWRITTEN DIGIT RECOGNITION USING NEURAL NETWORKS")
    print("#" * 70)
    
    p1_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "Project1_Handwritten_Digit_Recognition", "src"))
    sys.path.insert(0, p1_src)

    from train import main as train_p1
    from evaluate import evaluate_models
    from compare_ml import compare_ml_and_dl

    print("\n>>> Step 1.1: Training DNN (TensorFlow/Keras & PyTorch)...")
    train_p1()

    print("\n>>> Step 1.2: Evaluating Models & Generating Visualizations...")
    evaluate_models()

    print("\n>>> Step 1.3: Comparing Traditional ML & Deep Learning Baselines...")
    compare_ml_and_dl()

    sys.path.pop(0)


def run_project2():
    print("\n" + "#" * 70)
    print("# PROJECT 2: Q-LEARNING BASED MAZE SOLVER")
    print("#" * 70)

    p2_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "Project2_QLearning_Maze_Solver", "src"))
    sys.path.insert(0, p2_src)

    from train import main as train_p2
    from evaluate import evaluate_maze
    from compare_hyperparams import run_hyperparameter_experiments

    print("\n>>> Step 2.1: Training Q-Learning Agent on Maze Environment...")
    train_p2()

    print("\n>>> Step 2.2: Evaluating Policy & Generating Maze Solution Path...")
    evaluate_maze()

    print("\n>>> Step 2.3: Hyperparameter Tuning Experiments (Alpha, Gamma, Epsilon)...")
    run_hyperparameter_experiments()

    sys.path.pop(0)


def main():
    total_start = time.time()
    print("=" * 70)
    print(" COER UNIVERSITY, ROORKEE - MCA III SEMESTER")
    print(" LAB SHEET-08: MINI PROJECTS ON DEEP LEARNING & REINFORCEMENT LEARNING")
    print("=" * 70)

    run_project1()
    run_project2()

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f" ALL EXPERIMENTS & MODELS EXECUTED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print(" Trained models saved in:   models/")
    print(" Visualizations saved in:   outputs/")
    print(" Comprehensive docs in:     docs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
