using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using MathNet.Numerics.LinearAlgebra;

namespace Alex
{
    class Layer
    {
        public Vector<double> output; 
        public Matrix<double> weights;
        public Vector<double> bias;
        public Vector<double> input;
        public int nodes;

        
        public Layer(int nodes, List<double> input)
        {
            this.nodes = nodes;
            output = Vector<double>.Build.Dense(nodes);
            weights = Matrix<double>.Build.Dense(nodes, input.Count);
            bias = Vector<double>.Build.Dense(nodes);
            this.input = Vector<double>.Build.Dense(input.Count);
            for (int i = 0; i < input.Count; i++)
            {
                this.input[i] = input[i];
                for (int j = 0; j < nodes; j++)
                {
                    weights[j, i] = (double) np.random.stardard_normal() * np.sqrt(2.0 / (double) input.Count);
                }
            }
            FeedForward();
        }
        public List<double> Output()
        {
            List<double> output = new List<double>();
            double[] outpu = this.output.ToArray();
            foreach (double outp in outpu)
                output.Add(outp);
            return output;
        }
        public void Update_Inputs(List<double> input)
        {
            for(int i = 0; i < input.Count; i++)
            {
                this.input[i] = input[i];
            }
        }

        public void FeedForward()
        {
            output = weights.Multiply(input);
            output = output.Add(bias);
            for (int i = 0; i < output.Count; i++)
                output[i] = Training.ReLU(output[i]);
        }
    }
    class OutputLayer : Layer
    {
        public double output_d { get; }
        public OutputLayer(int outputs, List<double> input) : base(outputs, input) 
        {
            output_d = Output()[0];
        }
        public void FeedFoward()
        {
            output = weights.Multiply(input);
            output = output.Add(bias);
        }
    }
}
