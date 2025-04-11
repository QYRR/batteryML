#include <stdio.h>
#include <stdlib.h>


// #define N_length 50
// #define N_feature 9
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
    // test data: the first 5 rows in "randomized/train.csv"
    float voltage[5] = {
        4.117000e+00,
        4.091000e+00,
        4.077000e+00,
        4.068000e+00,
        4.061000e+00,
    };
    float current[5] = {
        1.004000e+00,
        9.990000e-01,
        1.000000e+00,
        1.000000e+00,
        9.990000e-01,
    };
    float temperature[5] = {
        2.151550e+01,
        2.157779e+01,
        2.159336e+01,
        2.164008e+01,
        2.170238e+01,
    };
    float time[5] = {
        4.000000e-02,
        3.004000e+01,
        6.004000e+01,
        9.004000e+01,
        1.200400e+02,
    };

    int N = 5;
    float out_features[9] = {};

    // test features output
    compute_features(voltage, current, temperature, time, N, out_features);
    printf("output in python: \t[-4.66666667e-04, 4.08280000e+00, 2.16058220e+01, \n\
    4.08446320e+00, 6.00000000e+01, 4.08115155e+00, \n\
    4.06100000e+00, 1.96574656e+00, 3.12156078e+00]\n");
    printf("output in C: \t\t[%.8e, %.8e, %.8e, \n%.8e, %.8e, %.8e, \n%.8e, %.8e, %.8e]\n",
            out_features[0], out_features[1], out_features[2], 
            out_features[3], out_features[4], out_features[5],
            out_features[6], out_features[7], out_features[8]);

    // test delta_current
    float* delta_current = (float*)malloc((N-1)*sizeof(float));    
    cumulative_trapezoid(time, current, delta_current, N);
    printf("delta_current in python: [30.045,  60.03 ,  90.03 , 120.015]\n");
    printf("delta_current in C: \t[%.8f, %.8f, %.8f, %.8f]\n", 
        delta_current[0], delta_current[1], delta_current[2], delta_current[3]);

    for(int i=0; i<N-1; i++)
    {
        delta_current[i] = delta_current[i] / 3600.0 * -1.0;
    }    

    // test dQ
    float* dQ = (float*)malloc((N-1)*sizeof(float));
    compute_gradient(delta_current, dQ, N-1);
    printf("dQ in python: \t[-0.00832917, -0.00833125, -0.00833125, -0.00832917]\n");
    printf("dQ in C: \t[%.8f, %.8f, %.8f, %.8f]\n", 
        dQ[0], dQ[1], dQ[2], dQ[3]);

    // test dV
    float* dV = (float*)malloc((N-1)*sizeof(float));
    compute_gradient(voltage, dV, N);
    printf("dV in python: \t[-0.026 , -0.02  , -0.0115, -0.008 ]\n");
    printf("dV in C: \t[%.8f, %.8f, %.8f, %.8f]\n", 
        dV[0], dV[1], dV[2], dV[3]);
}


void compute_features(float* voltage, float* current, float* temperature, float* time, 
    int N, float* out_features)
{
	/*
    'mean_discharge_voltage_rate', 'mean_voltage', 
    'mean_temperature', 'mean_power', 
    'mean_relativeTime', 'mean_resistance', 
    'min_voltage', 'mean_dV_dQ', 'max_dV_dQ'
    */

    float sum_discharge_voltage_rate = 0;
    float sum_voltage = voltage[0];
    float sum_temperature = temperature[0];
    float sum_power = voltage[0] * current[0];
    float sum_relativeTime = 0;
    float sum_resistance = voltage[0] / (current[0] + 1e-9f); 
    float min_voltage = voltage[0];
    float sum_dV_dQ = 0;
    float max_dV_dQ = -3.4028235e+38f;   // the min number that can be represented by float type

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
        sum_temperature += temperature[i];
        sum_power += voltage[i] * current[i];
        sum_relativeTime += time[i];
        sum_resistance += voltage[i] / (current[i] + 1e-9f);
        if (voltage[i] < min_voltage) {min_voltage = voltage[i];}
        float dV_dQ = dV[i-1] / dQ[i-1];
        sum_dV_dQ += dV_dQ;
        if (dV_dQ > max_dV_dQ) {max_dV_dQ = dV_dQ;}
    }

    float mean_discharge_voltage_rate = sum_discharge_voltage_rate / (N-1);
    float mean_voltage = sum_voltage / N;
    float mean_temperature = sum_temperature / N;
    float mean_power = sum_power / N; //
    float mean_relativeTime = sum_relativeTime / N;
    float mean_resistance = sum_resistance / N; //
    float mean_dV_dQ = sum_dV_dQ / (N-1);

    out_features[0] = mean_discharge_voltage_rate;
    out_features[1] = mean_voltage;
    out_features[2] = mean_temperature;
    out_features[3] = mean_power;
    out_features[4] = mean_relativeTime;
    out_features[5] = mean_resistance;
    out_features[6] = min_voltage;
    out_features[7] = mean_dV_dQ;
    out_features[8] = max_dV_dQ; 
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


