

#include <ensemble.h>
#include <ensemble_data.h>
#include <input.h>

// Default is to use the ensemble_arrays function

void inference();

void ensemble_inference(uint8_t children_right[N_NODES], float alphas[N_NODES],
                        uint8_t features[N_NODES], uint16_t roots[N_TREES],
                        float input[INPUT_LENGTH], float output[OUTPUT_LENGTH]);

void ensemble_inference(uint8_t children_right[N_NODES], float alphas[N_NODES],
                        uint8_t features[N_NODES], uint16_t roots[N_TREES],
                        float input[INPUT_LENGTH],
                        float output[OUTPUT_LENGTH]) {

  for (int t = 0; t < N_TREES; t++) {

    int current_idx = roots[t];
    int current_feature = features[current_idx];
    while (current_feature != LEAF_INDICATOR) {
      if (input[current_feature] <=
          alphas[current_idx]) { // False(0) -> Right, True(1) -> Left
        current_idx++;
      } else {
        current_idx += children_right[current_idx];
      }
      current_feature = features[current_idx];
    }

    output[0] += alphas[current_idx];
  }
}

// Inference function that calls ensemble_inference, depends on the parameters
void inference() {
  ensemble_inference(CHILDREN_RIGHT, ALPHAS, FEATURES, ROOTS, INPUT, OUTPUT);
}

// Main function, init stuff , call "inference()" and then print
int main(int argc, char **argv) {
  inference();

  printf("Output:\n");
  printf("%f\n", OUTPUT[0]);
}
