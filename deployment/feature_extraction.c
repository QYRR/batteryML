#include <stdio.h>
#include <stdlib.h>


// #define N_length 50
// #define N_feature 6
// float voltage[N_length];
// float current[N_length];
// float temperature[N_length];
// float relativeTime[N_length];
// float out_features[N_feature];


void compute_features(float* voltage, float* current, float* temperature, float* time, 
    int N, float* out_features);
void compute_gradient(float* x, float* gradient, int length);
void cumulative_trapezoid(float* x, float* y, float* ct_array, int length);


void main()
{
    // test data: the first 5 rows in "train.csv"
    float voltage[5] = {
        4.18888138,
        3.97915670,
        3.95692418,
        3.94005384,
        3.92609560
    };
    float current[5] = {
        -8.79471266e-04,
        -2.01465403,
        -2.01370335,
        -2.01279705,
        -2.01127254
    };
    float temperature[5] = {
        2.46859482e+01,
        2.47381321e+01,
        2.48970708e+01,
        2.50847501e+01,
        2.52646293e+01
    };
    float time[5] = {
        1.66720000e+01,
        3.57030000e+01,
        5.38280000e+01,
        7.18910000e+01,
        9.00470000e+01
    };

    int N = 5;
    float out_features[6] = {};

    // test features output
    compute_features(voltage, current, temperature, time, N, out_features);
    printf("output in python: \t[-3.48738749e-03, 3.99822234e+00, 7.95219148e-03, -6.36307131e+00, 3.69562000e+01, -8.89760004e+00]\n");
    printf("output in C: \t\t[%.8e, %.8e, %.8e, %.8e, %.8e, %.8e]\n",
            out_features[0], out_features[1], out_features[2], 
            out_features[3], out_features[4], out_features[5]);

    // test delta_current
    float* delta_current = (float*)malloc((N-1)*sizeof(float));    
    cumulative_trapezoid(time, current, delta_current, N);
    printf("delta_current in python: [-19.17880904, -55.68579777, -92.0511361, -128.58163984]\n");
    printf("delta_current in C: \t[%.8f, %.8f, %.8f, %.8f]\n", 
        delta_current[0], delta_current[1], delta_current[2], delta_current[3]);

    for(int i=0; i<N-1; i++)
    {
        delta_current[i] = delta_current[i] /3600.0 * -1.0;
    }    

    // test dQ
    float* dQ = (float*)malloc((N-1)*sizeof(float));
    compute_gradient(delta_current, dQ, N-1);
    printf("dQ in python: \t[0.01014083, 0.01012116, 0.01012442, 0.01014736]\n");
    printf("dQ in C: \t[%.8f, %.8f, %.8f, %.8f]\n", 
        dQ[0], dQ[1], dQ[2], dQ[3]);

    // test dV
    float* dV = (float*)malloc((N-1)*sizeof(float));
    compute_gradient(voltage, dV, N);
    printf("dV in python: \t[-0.20972468, -0.1159786 , -0.01955143, -0.01541429]\n");
    printf("dV in C: \t[%.8f, %.8f, %.8f, %.8f]\n", 
        dV[0], dV[1], dV[2], dV[3]);
}


void compute_features(float* voltage, float* current, float* temperature, float* time, 
    int N, float* out_features)
{
	/*
    'mean_discharge_voltage_rate', 'mean_voltage', 
    'mean_discharge_temperature_rate', 'mean_power', 
    'mean_relativeTime', 'mean_dV_dQ'
    */

    float sum_discharge_voltage_rate = 0;
    float sum_voltage = voltage[0];
    float sum_discharge_temperature_rate = 0;
    float sum_power = voltage[0] * current[0];
    float sum_relativeTime = 0;
    float sum_dV_dQ = 0;

    // set the reference of 'time' as 0
    float time_0 = time[0];
    for(int i=0; i<N; i++)
    {
        time[i] -= time_0;
    }

    // dQ
    // our device doesn't support 'malloc': 
    // float* delta_current = (float*)malloc((N-1)*sizeof(float));
    float delta_current[N];
    float dQ[N];
    cumulative_trapezoid(time, current, delta_current, N);
    for(int i=0; i<N-1; i++)
    {
        delta_current[i] = delta_current[i] / 3600.0 * -1.0;
    }    
    compute_gradient(delta_current, dQ, N-1);

    //dV
    float dV[N];
    compute_gradient(voltage, dV, N);


    for (int i=1; i<N; i++) {
        sum_discharge_voltage_rate += (voltage[i]-voltage[i-1]) / (time[i]-time[i-1]);
    	sum_voltage += voltage[i];
        sum_discharge_temperature_rate += (temperature[i]-temperature[i-1]) / (time[i]-time[i-1]);
        sum_power += voltage[i] * current[i];
        sum_relativeTime += time[i];
        sum_dV_dQ += dV[i-1] / dQ[i-1];
    }

    float mean_discharge_voltage_rate = sum_discharge_voltage_rate / (N-1);
    float mean_voltage = sum_voltage / N;
    float mean_discharge_temperature_rate = sum_discharge_temperature_rate / (N-1);
    float mean_power = sum_power / N;
    float mean_relativeTime = sum_relativeTime / N;
    float mean_dV_dQ = sum_dV_dQ / (N-1);

    out_features[0] = mean_discharge_voltage_rate;
    out_features[1] = mean_voltage;
    out_features[2] = mean_discharge_temperature_rate;
    out_features[3] = mean_power;
    out_features[4] = mean_relativeTime;
    out_features[5] = mean_dV_dQ;    
}


void compute_gradient(float* x, float* gradient, int length)
{
    gradient[0] = x[1] - x[0];
    gradient[length-1] = x[length-1] - x[length-2];
    for(int i=1; i<length-1; i++)
    {
        gradient[i] = (x[i+1]-x[i-1])/2.0f;
    }
}

void cumulative_trapezoid(float* x, float* y, float* ct_array, int length)
{
    float ct = 0;
    for(int i=0; i<length-1; i++)
    {
        float t = (y[i]+y[i+1])*(x[i+1]-x[i])/2.0f;
        ct += t;
        ct_array[i] = ct;
    }
}
