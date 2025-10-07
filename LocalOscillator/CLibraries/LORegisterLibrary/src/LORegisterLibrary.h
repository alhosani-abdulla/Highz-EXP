#pragma once
#include <Arduino.h>
#include <math.h>

void calcRegs(double freq, int *INT, int *MOD, int *FRAC, int *outputDiv, double *bandSelectClockDivider);

void makeRegs(int INT, int FRAC, int MOD, int outputDiv, int bandSelectClockDiv, uint32_t *reg0, uint32_t *reg1, uint32_t *reg2,
    uint32_t *reg3, uint32_t *reg4, uint32_t *reg5);