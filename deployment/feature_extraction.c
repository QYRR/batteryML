void compute_features(float* voltage, float* current, float* temperature, float* time, 
    int N, float* out_features);
void compute_gradient(float* x, float* gradient, int length);
void cumulative_trapezoid(float* x, float* y, float* ct_array, int length);


void compute_features(float* voltage, float* current, float* temperature, float* time, 
    int N, float* out_features)
{
	/*
    'mean_discharge_voltage_rate', 'mean_voltage', 
    'mean_discharge_temperature_rate', 'mean_power', 
    'mean_relativeTime', 'mean_dV_dQ', 
    'max_dV_dQ', 'min_dV_dQ', 'duration',
    'mean_temperature'
    */

    float sum_discharge_voltage_rate = 0;
    float sum_voltage = voltage[0];
    float sum_discharge_temperature_rate = 0;
    float sum_power = voltage[0] * current[0];
    float sum_relativeTime = 0;
    float sum_dV_dQ = 0;
    float max_dV_dQ = -3.4028235e+38f;  // the min number that can be represented by float type
    float min_dV_dQ = 3.4028235e+38f;  // the max number that can be represented by float type
    float duration = 0;     // = max_relativetime - min_relativetime
    float sum_temperature = temperature[0];  
    
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
        float dV_dQ = dV[i-1] / dQ[i-1];
        sum_dV_dQ += dV_dQ;
        if (dV_dQ > max_dV_dQ) {max_dV_dQ = dV_dQ;}
        if (dV_dQ < min_dV_dQ) {min_dV_dQ = dV_dQ;}
        sum_temperature += temperature[i];
    }

    float mean_discharge_voltage_rate = sum_discharge_voltage_rate / (N-1);
    float mean_voltage = sum_voltage / N;
    float mean_discharge_temperature_rate = sum_discharge_temperature_rate / (N-1);
    float mean_power = sum_power / N; //
    float mean_relativeTime = sum_relativeTime / N;
    float mean_dV_dQ = sum_dV_dQ / (N-1);
    duration = time[N-1];
    float mean_temperature = sum_temperature / N;

    out_features[0] = mean_discharge_voltage_rate;
    out_features[1] = mean_voltage;
    out_features[2] = mean_discharge_temperature_rate;
    out_features[3] = mean_power;
    out_features[4] = mean_relativeTime;
    out_features[5] = mean_dV_dQ;
    out_features[6] = max_dV_dQ;
    out_features[7] = min_dV_dQ;
    out_features[8] = duration;
    out_features[9] = mean_temperature;
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


