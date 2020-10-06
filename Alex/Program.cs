using System;
using System.IO;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Threading;
using YahooFinanceApi;
using NumSharp;
using MathNet.Numerics.LinearAlgebra; 

namespace Alex
{

    public class Program
    {

        static async Task Main()
        {
            const int hidden_layer_nodes = 32;
            const double lr = 1e-8;

            List<Training_Instance> ti = new List<Training_Instance>();

            //adds the training instances. they are not initialized.
            foreach (string tinker in Training.DATA)
            {
                for(int i = 1; i < 32; i++)
                    ti.Add(new Training_Instance(tinker, new DateTime(2019, 1, i)));
                for (int i = 1; i < 29; i++)
                    ti.Add(new Training_Instance(tinker, new DateTime(2019, 2, i)));
                for (int i = 1; i < 32; i++)
                    ti.Add(new Training_Instance(tinker, new DateTime(2019, 3, i)));
            }

            //foreach (string tinker in Training.DATA)
            //{
            //    for (DateTime dt = new DateTime(2010, 1, 1); !dt.Equals(new DateTime(2010, 1, 15)); dt.AddDays(1))
            //    {
            //        ti.Add(new Training_Instance(tinker, dt));
            //    }
            //}

            //initializing training instances
            foreach (Training_Instance ta in ti)
            {

                var history = await Yahoo.GetHistoricalAsync(ta.tinker, ta.dt.AddDays(-30.0), ta.dt, Period.Daily);
                double[] highs = new double[20];
                double[] lows = new double[20];
                double[] open = new double[20];
                int counter = 0;
                foreach (var candle in history)
                {
                    //add whatever input nessesary here
                    highs[counter] = (double)candle.High;
                    //lows[counter] = (double)candle.Low;
                    //open[counter] = (double)candle.Open;
                    counter++;
                    if (counter >= 17)
                        break;
                }
                for (int i = 0; i < 16; i++)
                {
                    ta.inputs.Add(highs[i]);
                }
                ta.target = highs[16];
            }

            //Initialize neural network
            Layer[] neuralNetwork = new Layer[5];
            neuralNetwork[0] = new Layer(hidden_layer_nodes, ti[0].inputs);
            neuralNetwork[1] = new Layer(hidden_layer_nodes, neuralNetwork[0].Output());
            neuralNetwork[2] = new Layer(hidden_layer_nodes, neuralNetwork[1].Output());
            neuralNetwork[3] = new Layer(hidden_layer_nodes, neuralNetwork[2].Output());
            neuralNetwork[4] = new OutputLayer(1, neuralNetwork[3].Output());

            var apr1 = await Yahoo.GetHistoricalAsync("AAPL", new DateTime(2019, 4, 1));

            for (int i = 0; i < 1000; i++)
            {
                Gradient_Descent();

                neuralNetwork[0].Update_Inputs(new Training_Instance("AAPL", new DateTime(2019, 4, 1)).inputs);
                neuralNetwork[0].FeedForward();
                neuralNetwork[1].Update_Inputs(neuralNetwork[0].Output());
                neuralNetwork[1].FeedForward();
                neuralNetwork[2].Update_Inputs(neuralNetwork[1].Output());
                neuralNetwork[2].FeedForward();
                neuralNetwork[3].Update_Inputs(neuralNetwork[2].Output());
                neuralNetwork[3].FeedForward();
                neuralNetwork[4].Update_Inputs(neuralNetwork[3].Output());
                ((OutputLayer)neuralNetwork[4]).FeedForward();
                Console.WriteLine("Apr 1: " + neuralNetwork[4].output[0] + " actual: " + (double)apr1[0].High);

            }

            var past = await Yahoo.GetHistoricalAsync(Training.DATA[0], DateTime.Now.AddDays(-21), DateTime.Now, Period.Daily);
            List<double> highss = new List<double>();
            double output = 0;
            List<double> layeroutput = new List<double>();

            foreach (var candle in past)
            {
                highss.Add((double)candle.High);
            }

            for (int j = 0; j < 30; j++)
            {
                neuralNetwork[0].Update_Inputs(highss);
                for (int i = 0; i < 4; i++)
                {
                    neuralNetwork[i].FeedForward();
                    neuralNetwork[i + 1].Update_Inputs(neuralNetwork[i].Output());
                }
                neuralNetwork[4].FeedForward();
                output = neuralNetwork[4].output[0];

                highss.RemoveAt(0);
                highss.Add(output);

                File.AppendAllText(@"C:\Users\asdfl\OneDrive\Desktop\AlexOut\Future.txt", output.ToString() + "\n");
            }

            highss.RemoveRange(0, highss.Count);
            past = await Yahoo.GetHistoricalAsync(Training.DATA[0], new DateTime(2019, 5, 1), new DateTime(2019, 5, 31), Period.Daily);

            foreach (var candle in past)
            {
                File.AppendAllText(@"C:\Users\asdfl\OneDrive\Desktop\AlexOut\May2019Actual.txt", candle.High + "\n");
            }

            past = await Yahoo.GetHistoricalAsync(Training.DATA[0], new DateTime(2019, 4, 5), new DateTime(2019, 4, 30), Period.Daily);

            foreach (var candle in past)
            {
                highss.Add((double)candle.High);
            }

            for (int j = 0; j < 21; j++)
            {
                neuralNetwork[0].Update_Inputs(highss);
                for (int i = 0; i < 4; i++)
                {
                    neuralNetwork[i].FeedForward();
                    neuralNetwork[i + 1].Update_Inputs(neuralNetwork[i].Output());
                }
                neuralNetwork[4].FeedForward();
                output = neuralNetwork[4].output[0];

                highss.RemoveAt(0);
                highss.Add(output);

                File.AppendAllText(@"C:\Users\asdfl\OneDrive\Desktop\AlexOut\May2019Predict.txt", output.ToString() + "\n");
            }

            void Gradient_Descent()
            {
                //dummy for total cost
                double tcost = 0.0;

                //Changes we make to the neural network weights and biases
                //Matrix<double>[] nn4w = Matrix<double>.Build.Dense(neuralNetwork[4].Output().Count, neuralNetwork[3].input.Count);
                //Matrix<double> nn3w = Matrix<double>.Build.Dense(neuralNetwork[3].Output().Count, neuralNetwork[3].input.Count);
                //double nn4b = 0.0;
                //Vector<double> nn3b = Vector<double>.Build.Dense(neuralNetwork[3].nodes);

                Matrix<double>[] NthGradient = new Matrix<double>[5];
                Vector<double>[] NthBias = new Vector<double>[5];

                Matrix<double>[] nthGradient = new Matrix<double>[5];
                Vector<double>[] nthBias = new Vector<double>[5];

                Matrix<double> FirstGradient = Matrix<double>.Build.Dense(neuralNetwork[4].Output().Count, neuralNetwork[4].input.Count);
                

                for (int i1 = 0; i1 < 5; i1++)
                {
                    NthGradient[i1] = Matrix<double>.Build.Dense(neuralNetwork[i1].output.Count, neuralNetwork[i1].input.Count, Matrix<double>.Zero);
                    NthBias[i1] = Vector<double>.Build.Dense(neuralNetwork[i1].nodes, Vector<double>.Zero);
                }

                for (int k = 1; k <= ti.Count; k++)
                {
                    //adds costs of the instance to total
                    double costDerivative = neuralNetwork[4].Output()[0] - ti[k - 1].target;
                    tcost += Training.cost(neuralNetwork[4].Output()[0], ti[k - 1].target);
                    Vector<double>[] derivActivation = new Vector<double>[5];

                    for (int i = 0; i < FirstGradient.ColumnCount; i++)
                        FirstGradient[0, i] = neuralNetwork[3].Output()[i] * costDerivative;

                    

                    Vector<double> fme = Vector<double>.Build.Dense(neuralNetwork[4].weights.ColumnCount);
                    //derivActivation[4] = costDeriv;
                    for (int i = 0; i < neuralNetwork[4].weights.ColumnCount; i++)
                    {
                        fme[i] = neuralNetwork[4].weights[0, i];
                    }
                    derivActivation[4] = fme.Multiply(costDerivative);
                    for (int i1 = 3; i1 >= 0; i1--) //derivative of cost with respect to one output node
                    {

                        derivActivation[i1] = neuralNetwork[i1].weights.ConjugateTranspose().Multiply(derivActivation[i1 + 1].PointwiseMultiply(Training.derivReLU(neuralNetwork[i1].output)));
                    }


                    for (int i1 = 0; i1 < 4; i1++)
                    {
                        nthGradient[i1] = Matrix<double>.Build.Dense(neuralNetwork[i1].output.Count, neuralNetwork[i1].input.Count, Matrix<double>.Zero);
                        for (int i = 0; i < neuralNetwork[i1].input.Count; i++)
                        {
                            for (int j = 0; j < neuralNetwork[i1].output.Count; j++)
                            {
                                if (i1 == 3)
                                {
                                    int j1 = 0;
                                    nthGradient[i1][j, i] += neuralNetwork[i1].input[i] /*B*/ * neuralNetwork[i1 + 1].weights[j1, j] /*Wbc*/ * derivActivation[i1][j1];
                                }
                                else
                                {
                                    for (int j1 = 0; j1 < neuralNetwork[i1].input.Count; j1++)
                                        nthGradient[i1][j, i] += neuralNetwork[i1].input[i] /*B*/ * neuralNetwork[i1 + 1].weights[j1, j] /*Wbc*/ * derivActivation[i1][j1];
                                }
                            }
                        }
                        if (i1 < 4)
                            nthBias[i1] = Training.derivReLU(neuralNetwork[i1].output).PointwiseMultiply(derivActivation[i1 + 1]);
                    }
                    nthGradient[4] = FirstGradient;
                    nthBias[4] = Vector<double>.Build.Dense(1);
                    nthBias[4][0] = costDerivative;
                    //for (int i1 = 0; i1 < 4; i1++)
                    //{
                    //    for (int i = 0; i < neuralNetwork[3].input.Count; i++)
                    //    {
                    //        for (int j = 0; j < neuralNetwork[3].output.Count; j++)
                    //        {
                    //            SecondGradient[j, i] = neuralNetwork[3].input[i] /*B*/ * neuralNetwork[4].weights[0, j] /*Wbc*/ * derivActivation[3];
                    //        }
                    //    }
                    //}



                    //Matrix<double> FirstGradient = Matrix<double>.Build.Dense(neuralNetwork[4].Output().Count, neuralNetwork[4].input.Count);
                    //for (int i = 0; i < FirstGradient.ColumnCount; i++)
                    //    FirstGradient[0, i] = neuralNetwork[3].Output()[i] * costDerivative;
                    ////derivActivation[3] = neuralNetwork[3].weights.Multiply(Training.derivReLU(neuralNetwork[4].output));
                    ////derivActivation[2] = derivActivation[3].ElementWiseMultiply(Training.derivReLU(neuralNetwork[3].output))
                    //Matrix<double> SecondGradient = Matrix<double>.Build.Dense(neuralNetwork[3].Output().Count, neuralNetwork[3].input.Count);


                    //Matrix<double> ThirdGradient = Matrix<double>.Build.Dense(neuralNetwork[2].Output().Count, neuralNetwork[2].input.Count);
                    //ThirdGradient = SecondGradient.Multiply()
                    //for(int i = 0; i < neuralNetwork[2].input.Count; i++)
                    //{
                    //    for(int j = 0; j < neuralNetwork[2].input.Count; j++)
                    //    {
                    //        //is this one right?
                    //        ThirdGradient[j, i] = neuralNetwork[2].input[i] * neuralNetwork[3].weights[0,j] * Training.derivReLU(neuralNetwork[2].output[j]) * costDerivative;
                    //    }
                    //}

                    //Matrix<double> FourthGradient = Matrix<double>.Build.Dense(neuralNetwork[1].Output().Count, neuralNetwork[2].input.Count);
                    //for (int i = 0; i < neuralNetwork[1].input.Count; i++)
                    //{
                    //    for (int j = 0; j < neuralNetwork[1].output.Count; j++)
                    //    {
                    //        FourthGradient[j, i] =
                    //    }
                    //}

                    //Vector<double> bias1 = Vector<double>.Build.Dense(neuralNetwork[3].nodes);
                    //for (int i = 0; i < neuralNetwork[3].nodes; i++)
                    //{
                    //    bias1[i] = neuralNetwork[4].weights[0, i] * Training.derivReLU(neuralNetwork[4].output[3]);

                    //}

                    //nn4w = nn4w.Add(FirstGradient.Multiply(lr));
                    //nn3w = nn3w.Add(SecondGradient.Multiply(lr));
                    //nn4b += costDerivative * lr;
                    //nn3b = nn3b.Add(bias1.Multiply(costDerivative * lr));

                    for (int i1 = 0; i1 <= 4; i1++)
                    {
                        NthGradient[i1] = NthGradient[i1].Add(nthGradient[i1].Multiply(lr));
                        NthBias[i1] = NthBias[i1].Add(nthBias[i1].Multiply(lr));
                    }

                    //updates the inputs of neural network
                    if (k != ti.Count)
                    {
                        neuralNetwork[0].Update_Inputs(ti[k].inputs);
                        neuralNetwork[0].FeedForward();
                        neuralNetwork[1].Update_Inputs(neuralNetwork[0].Output());
                        neuralNetwork[1].FeedForward();
                        neuralNetwork[2].Update_Inputs(neuralNetwork[1].Output());
                        neuralNetwork[2].FeedForward();
                        neuralNetwork[3].Update_Inputs(neuralNetwork[2].Output());
                        neuralNetwork[3].FeedForward();
                        neuralNetwork[4].Update_Inputs(neuralNetwork[3].Output());
                        ((OutputLayer)neuralNetwork[4]).FeedForward();
                    }
                }

                //modifying values
                for(int i1 = 4; i1>=0; i1--)
                {
                    neuralNetwork[i1].weights = neuralNetwork[i1].weights.Subtract(NthGradient[i1]);
                    neuralNetwork[i1].bias = neuralNetwork[i1].bias.Subtract(NthBias[i1]);
                }
                //neuralNetwork[4].weights = neuralNetwork[4].weights.Subtract(nthGradient[4]);
                //neuralNetwork[3].weights = neuralNetwork[3].weights.Subtract(nn3w);
                //neuralNetwork[4].bias = neuralNetwork[4].bias.Subtract(nn4b);
                //neuralNetwork[3].bias = neuralNetwork[3].bias.Subtract(nn3b);
                neuralNetwork[0].Update_Inputs(ti[0].inputs);
                for (int i1 = 0; i1 < 4; i1++)
                {
                    neuralNetwork[i1].FeedForward();
                    neuralNetwork[i1 + 1].Update_Inputs(neuralNetwork[i1].Output());
                }
                ((OutputLayer) neuralNetwork[4]).FeedForward();
                Console.WriteLine("Cost: " + tcost);
                //Console.WriteLine("Output: " + neuralNetwork[4].output[0]);
            }
        }
    }
}