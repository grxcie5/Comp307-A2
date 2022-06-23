import java.util.Arrays;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;
    public final double[] hiddenBiases;
    public final double[] outputBiases;
    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;

    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights, double[][] initial_output_layer_weights, double learning_rate, double[] hiddenBiases, double[] outputBiases) {
        //Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;
        this.hiddenBiases = hiddenBiases;
        this.outputBiases = outputBiases;

        this.learning_rate = learning_rate;
    }


    /**
     * Uses Sigmoid as an activation function.
     * y = 1 / (1 + e^-z)
     * @param input
     * @return sigmoid applied to input.
     */
    public double sigmoid(double input) {

        double output = 1.0 / (1.0 + Math.exp(-1 * input));
        return output;
    }

    /**
     * Feed forward pass input to a network output.
     * Calculated the hidden layer and output layer output values.
     * @param inputs
     * @return 2d array of hidden layer outputs, output layer outputs.
     */
    public double[][] forward_pass(double[] inputs) {
        double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            for( int j = 0; j < inputs.length; j++){
                weighted_sum += (inputs[j] * hidden_layer_weights[j][i]);
            }
            weighted_sum = hiddenBiases[i] + weighted_sum; //add bias to the weighted sum
            double output = sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            for( int j = 0; j < num_hidden; j++){
                weighted_sum += hidden_layer_outputs[j] * output_layer_weights[j][i];
            }
            weighted_sum = outputBiases[i] + weighted_sum;
            double output = sigmoid(weighted_sum);
            output_layer_outputs[i] = output;
        }

        return new double[][]{hidden_layer_outputs, output_layer_outputs};
    }

    /**
     * Calculates the betas then the change in weights and biases.
     * @param inputs
     * @param hidden_layer_outputs
     * @param output_layer_outputs
     * @param desired_outputs
     * @return
     */
    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs,
                                                 double[] output_layer_outputs, int desired_outputs) {

        double[] output_layer_betas = new double[num_outputs];
        // TODO! Calculate output layer betas.
        for (int i=0; i<num_outputs; i++){
            if (i == desired_outputs){
                output_layer_betas[i] = 1 - output_layer_outputs[i];
            } else {
                output_layer_betas[i] = 0 - output_layer_outputs[i];
            }
        }

        double[] hidden_layer_betas = new double[num_hidden];
        // TODO! Calculate hidden layer betas.
        for(int i=0; i<num_hidden; i++){
            double beta = 0;
            for(int j=0; j<num_outputs; j++){
                beta += output_layer_weights[i][j] * output_layer_outputs[j]
                        * (1 - output_layer_outputs[j]) * output_layer_betas[j];
            }
            hidden_layer_betas[i] = beta;
        }

        // This is a HxO array (H hidden nodes, O outputs)
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];
        // TODO! Calculate output layer weight changes.
        for(int i=0; i<num_hidden; i++){
            for(int j=0; j<num_outputs; j++){
                double oIn = hidden_layer_outputs[i];
                double oOut = output_layer_outputs[j];
                double beta = output_layer_betas[j];
                delta_output_layer_weights[i][j] = this.learning_rate * oIn * oOut * (1 - oOut) * beta;
            }
        }

        // This is a IxH array (I inputs, H hidden nodes)
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
        // TODO! Calculate hidden layer weight changes.
        for(int i=0; i<num_inputs; i++){
            for(int j=0; j<num_hidden; j++){
                double oIn = inputs[i];
                double oHid = hidden_layer_outputs[j];
                double beta = hidden_layer_betas[j];
                delta_hidden_layer_weights[i][j] = this.learning_rate * oIn * oHid * (1 - oHid) * beta;
            }
        }

        //calculating the change in hidden layer biases
        double[] delta_hiddenBias = new double[num_hidden];
        for(int i=0; i<num_hidden; i++){
            double delta = this.learning_rate * hidden_layer_outputs[i] * (1 - hidden_layer_outputs[i]) * hidden_layer_betas[i];
            delta_hiddenBias[i] = delta;
        }

        //calculating the change in output layer biases
        double[] delta_outputBias = new double[num_outputs];
        for(int i=0; i<num_outputs; i++){
            double delta = this.learning_rate * output_layer_outputs[i] * (1 - output_layer_outputs[i]) * output_layer_betas[i];
            delta_outputBias[i] = delta;
        }

        // Return the weights we calculated, so they can be used to update all the weights.
        //store delta hidden and output biases in a 2d array to return.
        double[][] deltaBiases = new double[][]{delta_hiddenBias, delta_outputBias};
        return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights, deltaBiases};
    }

    /**
     * Updates the output layer and hidden layer weights.
     * Updates the output layer and hidden layer bias nodes.s
     * @param delta_output_layer_weights
     * @param delta_hidden_layer_weights
     * @param deltaBiases
     */
    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights, double[][] deltaBiases) {
        // TODO! Update the weights
        //changing the hidden layer weights
        for(int i=0; i<num_inputs; i++){
            for(int j=0; j<num_hidden; j++){
                hidden_layer_weights[i][j] += delta_hidden_layer_weights[i][j];
            }
        }

        //changing the output layer weights
        for(int i=0; i<num_hidden; i++){
            for(int j=0; j<num_outputs; j++){
                output_layer_weights[i][j] += delta_output_layer_weights[i][j];
            }
        }

        double[] deltaHiddenBias = deltaBiases[0];
        double[] deltaOutputBias = deltaBiases[1];

        //update the hidden layer bias nodes
        for(int i=0; i<num_hidden; i++){
            hiddenBiases[i] += deltaHiddenBias[i];
        }

        //update the output layer bias nodes
        for(int i=0; i<num_outputs; i++){
            outputBiases[i] += deltaOutputBias[i];
        }
    }

    /**
     * For all instances, performs a feed forward pass, a backpropagation, predicts the class
     * then updates the weights for a number of epochs.
     * @param instances
     * @param desired_outputs
     * @param epochs
     */
    public void train(double[][] instances, int[] desired_outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("Hidden layer weights: \n" + Arrays.deepToString(hidden_layer_weights));
            System.out.println("Output layer weights: \n" + Arrays.deepToString(output_layer_weights));
            double count =0;
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i]);
                //gets the predicted class of an instance
                int predicted_class = predict(instance); // TODO!
                //if it's the same as the desired output, count it as a correct prediction for this epoch
                if(predicted_class == desired_outputs[i]){
                    count++;
                }
                update_weights(delta_weights[0], delta_weights[1], delta_weights[2]);
            }
            // TODO: Print accuracy achieved over this epoch
            double acc = (count / instances.length);
            System.out.println("EPOCH = " + epoch + "   ACCURACY = " + acc + "\n");
        }
    }

    /**
     * Predicts a class for one instance at a time.
     * @param instances
     * @return the predicted class of one instance
     */
    public int predict(double[] instances) {
        int predicted_class = 0;
            double[][] outputs = forward_pass(instances);
            //goes through output layer outputs and finds the index of the largest value as the predicted class
            double max = 0;
            for( int j=0; j<outputs[1].length; j++){
                if(outputs[1][j] > max){
                    max = outputs[1][j];
                    predicted_class = j;
                }
            }
        return predicted_class;
    }

    /**
     * Predicts the class for the first instance. Performs a feedforward
     * pass and prints off the resulting output values.
     * for the first instance.
     * @param instances
     * @return an int array of the predicted classes.
     */
    public int[] predictFirstInstance(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            int predicted_class = -1;  // TODO !Should be 0, 1, or 2.
            double max = Double.MIN_VALUE;
            System.out.println("first instance output values: " + Arrays.toString(outputs[1]));
            //goes through output layer outputs and finds the index of the largest value as the predicted class
            for( int j=0; j<outputs[1].length; j++){
                if(outputs[1][j] > max){

                    max = outputs[1][j];
                    predicted_class = j;
                }
            }
            System.out.println("predicted class at: " + predicted_class);
            predictions[i] = predicted_class;
        }
        return predictions;
    }

    /**
     * Predicts the classes for the whole test set.
     * @param instances test instances
     * @return an array of predicted classes for the whole test set.
     */
    public int[] predictTest(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            int predicted_class = -1;  // TODO !Should be 0, 1, or 2.
            double max = Double.MIN_VALUE;
            for( int j=0; j<outputs[1].length; j++){
                if(outputs[1][j] > max){
                    max = outputs[1][j];
                    predicted_class = j;
                }
            }
            predictions[i] = predicted_class;
        }
        return predictions;
    }

}
