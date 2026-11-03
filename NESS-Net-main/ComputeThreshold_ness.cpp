#include "NAM.h"

void ComputeThreshold(const list<float> &Dist, const float &Percent, float &THRESHOLD) {
    if (Dist.empty()) {
        THRESHOLD = 0;
        return;
    }
    vector<float> values;
    for (float value : Dist) {
        values.push_back(value + 1e-5f);
    }
    sort(values.begin(), values.end(), greater<float>{});
    int index = static_cast<int>(floor((1 - Percent / 100.0f) * Dist.size())) - 1;
    index = max(0, min(index, static_cast<int>(values.size()) - 1));
    THRESHOLD = values[index];
}
