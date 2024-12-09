

import argparse
import random


# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        trans_file = f"{basename}.trans"
        emit_file = f"{basename}.emit"

        self.transitions = {}
        self.emissions = {}

        with open(trans_file, "r") as tf:
            for line in tf:
                parts = line.strip().split()
                if parts[0] == "#":
                    state_from = parts[1]
                    prob = float(parts[2])
                    if "#" not in self.transitions.keys():
                        self.transitions["#"] = {}
                    self.transitions["#"][state_from] = prob
                else:
                    state_from, state_to, prob = parts
                    if state_from not in self.transitions.keys():
                        self.transitions[state_from] = {}
                    self.transitions[state_from][state_to] = float(prob)

        with open(emit_file, "r") as ef:
            for line in ef:
                state, observation, prob = line.strip().split()
                if state not in self.emissions:
                    self.emissions[state] = {}
                self.emissions[state][observation] = float(prob)

        print("Transitions:", self.transitions)
        print("Emissions:", self.emissions)


   ## you do this.

    def generate(self, n):
        state = "#"
        states = []
        outputs = []

        for _ in range(n):
            if state not in self.transitions.keys() or not self.transitions[state]:
                break

            next_state = random.choices(
                population=list(self.transitions[state].keys()),
                weights=list(self.transitions[state].values())
            )[0]
            states.append(next_state)

            if next_state in self.emissions:
                emission = random.choices(
                    population=list(self.emissions[next_state].keys()),
                    weights=list(self.emissions[next_state].values())
                )[0]
                outputs.append(emission)
            else:
                outputs.append(None)

            state = next_state

        return ' '.join(outputs)


    def forward(self, sequence):

        alpha = [{}]

        for state in self.transitions["#"]:
            initial_transmission = self.transitions["#"].get(state, 0)
            initial_emission = self.emissions[state].get(sequence[0], 0)
            alpha[0][state] = initial_transmission * initial_emission

        for t in range(1, len(sequence)):
            alpha.append({})
            for state in self.transitions:
                if state == "#":
                    continue
                transition_contributions = [
                    alpha[t - 1][prev_state] * self.transitions[prev_state].get(state, 0)
                    for prev_state in self.transitions if prev_state != "#"
                ]
                max_transition = sum(transition_contributions)
                emission_prob = self.emissions[state].get(sequence[t], 0)
                alpha[t][state] = max_transition * emission_prob
                print(f"t={t}, state={state}, transition_contributions={transition_contributions}, "
                      f"emission_prob={emission_prob}, alpha={alpha[t][state]}")

        final_probs = {state: alpha[-1][state] for state in alpha[-1]}
        most_probable_state = max(final_probs, key=final_probs.get)
        max_prob = max(final_probs.values())
        return most_probable_state, max_prob


    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.

    def viterbi(self, sequence):
        prob_table = [{}]
        paths = {}

        for start_state in self.transitions["#"]:
            init_prob = self.transitions["#"].get(start_state, 0) * self.emissions.get(start_state, {}).get(sequence[0], 0)
            prob_table[0][start_state] = init_prob
            paths[start_state] = [start_state]

        for t in range(1, len(sequence)):
            prob_table.append({})
            new_paths = {}

            states = [s for s in self.transitions.keys() if s != "#"]
            for current_state in states:
                max_prob = 0.0
                best_prev_state = None
                emit_prob = self.emissions.get(current_state, {}).get(sequence[t], 0)

                if emit_prob == 0:
                    continue

                for prev_state in prob_table[t - 1]:
                    if prev_state not in self.transitions:
                        continue
                    trans_prob = self.transitions.get(prev_state, {}).get(current_state, 0)
                    if trans_prob == 0:
                        continue
                    current_prob = prob_table[t - 1][prev_state] * trans_prob * emit_prob
                    if current_prob > max_prob:
                        max_prob = current_prob
                        best_prev_state = prev_state

                if best_prev_state is not None and max_prob > 0:
                    prob_table[t][current_state] = max_prob
                    new_paths[current_state] = paths[best_prev_state] + [current_state]

            paths = new_paths

        final_probs = prob_table[-1] if len(prob_table) > 0 else {}
        if not final_probs:
            return [], 0.0

        most_prob_final_state = max(final_probs, key=final_probs.get)
        highest_prob = final_probs[most_prob_final_state]
        most_prob_path = paths[most_prob_final_state]

        return most_prob_path, highest_prob


    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HMM")
    parser.add_argument("basename", type=str, help="Base name of model (<basename>.trans | <basename>.emit")
    parser.add_argument("--generate", type=int, help="Length of sequence to generate")
    parser.add_argument("--forward", type=str, help="sequence for forward algorithm")
    parser.add_argument("--viterbi", type=str, help="sequence for viterbi algorithm")
    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.basename)

    if args.generate:
        sequence = hmm.generate(args.generate)
        print(sequence)

    if args.forward:
        with open(args.forward, "r") as file:
            observations = file.read().strip().split()
        most_probable_state, prob = hmm.forward(args.forward)
        print(f"Most probable final state: {most_probable_state}, Probability: {prob}")

    if args.viterbi:
        with open(args.viterbi, "r") as file:
            lines = file.read().strip().split("\n")

        tagged_output = [] # this is to get the pretty thing so the language looks like the tagged.obs file :)
        for line in lines:
            if not line.strip():
                continue
            observations = line.strip().split()
            states, prob = hmm.viterbi(observations)
            tagged_output.append("States:\t\t\t" + " ".join(states))
            tagged_output.append("Observations:\t" + " ".join(observations))

        print("\n".join(tagged_output))
