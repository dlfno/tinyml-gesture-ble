#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class NBModel {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        float votes[4] = { 0.0f };
                        float theta[32] = { 0 };
                        float sigma[32] = { 0 };
                        theta[0] = -3.8680642; theta[1] = 0.005887142; theta[2] = -3.8742971; theta[3] = -3.8565357; theta[4] = 3.8680718; theta[5] = 0.20383607; theta[6] = 0.5514345; theta[7] = -0.6891681; theta[8] = 1.2029355; theta[9] = 0.6524203; theta[10] = -0.111030124; theta[11] = 0.3905891; theta[12] = -0.7957356; theta[13] = 0.53198767; theta[14] = 0.65246266; theta[15] = -0.79588366; theta[16] = 1147.261; theta[17] = -1959.169; theta[18] = 1960.5696; theta[19] = 1152.7285; theta[20] = -1.0960729; theta[21] = 1145.8099; theta[22] = -1962.264; theta[23] = 1957.6538; theta[24] = 1151.7793; theta[25] = 2.3054302; theta[26] = 1147.9631; theta[27] = -1961.5573; theta[28] = 1959.7224; theta[29] = 1153.902; theta[30] = 4.9616218; theta[31] = 2996.4487;
                        sigma[0] = 0.00012415933; sigma[1] = 0.00012609939; sigma[2] = 0.000103813116; sigma[3] = 0.00026011208; sigma[4] = 0.0001240897; sigma[5] = 0.092004806; sigma[6] = 0.10302843; sigma[7] = 0.44618183; sigma[8] = 0.5924321; sigma[9] = 0.11490753; sigma[10] = 0.45377675; sigma[11] = 0.06969401; sigma[12] = 1.2561604; sigma[13] = 0.27257407; sigma[14] = 0.26254845; sigma[15] = 12534.658; sigma[16] = 2925.5613; sigma[17] = 1560.3807; sigma[18] = 1507.0605; sigma[19] = 2891.3967; sigma[20] = 13662.76; sigma[21] = 2885.535; sigma[22] = 1488.826; sigma[23] = 1629.9092; sigma[24] = 2832.6326; sigma[25] = 13599.836; sigma[26] = 2695.2776; sigma[27] = 1449.0316; sigma[28] = 1364.1492; sigma[29] = 2631.4482; sigma[30] = 0.4133127; sigma[31] = 10412.351;
                        votes[0] = 0.31674477 - gauss(x, theta, sigma);
                        theta[0] = -3.8656285; theta[1] = 0.0075898785; theta[2] = -3.8745222; theta[3] = -3.8505785; theta[4] = 3.8656373; theta[5] = 0.097121365; theta[6] = 0.6038478; theta[7] = -1.1266661; theta[8] = 1.5249752; theta[9] = 0.8050365; theta[10] = -0.20301324; theta[11] = 0.5182257; theta[12] = -1.6366813; theta[13] = 0.89328015; theta[14] = 0.6739351; theta[15] = -6.9295335; theta[16] = 1145.3771; theta[17] = -1961.1495; theta[18] = 1962.171; theta[19] = 1151.0801; theta[20] = -0.7889019; theta[21] = 1144.9949; theta[22] = -1959.6904; theta[23] = 1956.4264; theta[24] = 1150.9948; theta[25] = -8.783625; theta[26] = 1146.1748; theta[27] = -1957.499; theta[28] = 1962.877; theta[29] = 1152.3269; theta[30] = 5.068472; theta[31] = 2989.4412;
                        sigma[0] = 0.00014963334; sigma[1] = 0.00012460604; sigma[2] = 0.00010671325; sigma[3] = 0.0002212889; sigma[4] = 0.00014955495; sigma[5] = 0.2898472; sigma[6] = 0.11500106; sigma[7] = 0.6926848; sigma[8] = 1.294236; sigma[9] = 0.1307274; sigma[10] = 0.15058725; sigma[11] = 0.07073878; sigma[12] = 0.8962256; sigma[13] = 0.57661635; sigma[14] = 0.07680716; sigma[15] = 13036.904; sigma[16] = 2821.695; sigma[17] = 1246.9008; sigma[18] = 1556.1049; sigma[19] = 2809.7883; sigma[20] = 13721.646; sigma[21] = 3183.8677; sigma[22] = 1612.8339; sigma[23] = 1839.2969; sigma[24] = 3130.5679; sigma[25] = 14002.358; sigma[26] = 3033.0999; sigma[27] = 1386.4888; sigma[28] = 1743.4126; sigma[29] = 2971.4104; sigma[30] = 0.14496475; sigma[31] = 10200.445;
                        votes[1] = 0.05066321 - gauss(x, theta, sigma);
                        theta[0] = -3.8690147; theta[1] = 0.0037655428; theta[2] = -3.8732448; theta[3] = -3.8597965; theta[4] = 3.8690417; theta[5] = -0.052341737; theta[6] = 0.6502327; theta[7] = -1.2193513; theta[8] = 1.080151; theta[9] = 0.7995695; theta[10] = -0.080034256; theta[11] = 0.27500504; theta[12] = -0.579362; theta[13] = 0.44978732; theta[14] = 0.51038414; theta[15] = 4.1373005; theta[16] = 1145.6743; theta[17] = -1958.4102; theta[18] = 1959.0376; theta[19] = 1151.4851; theta[20] = -0.80642587; theta[21] = 1149.8969; theta[22] = -1961.5448; theta[23] = 1960.6305; theta[24] = 1155.733; theta[25] = 1.6212012; theta[26] = 1147.3757; theta[27] = -1961.3046; theta[28] = 1960.1141; theta[29] = 1153.3845; theta[30] = 5.017159; theta[31] = 2999.7163;
                        sigma[0] = 0.00012766091; sigma[1] = 0.00028165855; sigma[2] = 0.000104558676; sigma[3] = 0.016284326; sigma[4] = 0.00012574365; sigma[5] = 0.18453775; sigma[6] = 0.16048196; sigma[7] = 0.8953449; sigma[8] = 0.5246479; sigma[9] = 0.13114715; sigma[10] = 0.23019043; sigma[11] = 0.028818946; sigma[12] = 0.44172832; sigma[13] = 0.19364263; sigma[14] = 0.08044776; sigma[15] = 13291.385; sigma[16] = 2645.8489; sigma[17] = 1546.4126; sigma[18] = 1472.0819; sigma[19] = 2602.9082; sigma[20] = 13387.44; sigma[21] = 2705.7312; sigma[22] = 1470.076; sigma[23] = 1579.5494; sigma[24] = 2637.4976; sigma[25] = 13771.216; sigma[26] = 2881.9072; sigma[27] = 1684.4071; sigma[28] = 1481.679; sigma[29] = 2828.162; sigma[30] = 0.25487405; sigma[31] = 10260.592;
                        votes[2] = 0.31634587 - gauss(x, theta, sigma);
                        theta[0] = -3.870344; theta[1] = 0.00037560938; theta[2] = -3.870835; theta[3] = -3.868847; theta[4] = 3.870347; theta[5] = 0.103034705; theta[6] = 0.015072582; theta[7] = 0.06733408; theta[8] = 0.13728745; theta[9] = 0.20770809; theta[10] = -0.20846593; theta[11] = 0.011233416; theta[12] = -0.23456652; theta[13] = -0.18282363; theta[14] = 0.46726313; theta[15] = -19.865547; theta[16] = 1060.8447; theta[17] = -1882.14; theta[18] = 1869.9275; theta[19] = 1125.6333; theta[20] = 20.009361; theta[21] = 1019.8959; theta[22] = -1807.9204; theta[23] = 1852.2893; theta[24] = 1149.1207; theta[25] = -51.762295; theta[26] = 1012.686; theta[27] = -1848.3524; theta[28] = 1784.4618; theta[29] = 1113.5173; theta[30] = 4.5426083; theta[31] = 2998.1387;
                        sigma[0] = 0.00015593726; sigma[1] = 0.00012548905; sigma[2] = 0.00014972787; sigma[3] = 0.0022286049; sigma[4] = 0.00015585474; sigma[5] = 0.099319205; sigma[6] = 0.00047099477; sigma[7] = 0.11071994; sigma[8] = 0.10134865; sigma[9] = 0.06738785; sigma[10] = 0.31603876; sigma[11] = 0.00051705306; sigma[12] = 0.3187718; sigma[13] = 0.31925276; sigma[14] = 0.14170274; sigma[15] = 121060.984; sigma[16] = 61828.094; sigma[17] = 84897.72; sigma[18] = 125542.72; sigma[19] = 41624.19; sigma[20] = 249015.52; sigma[21] = 100122.2; sigma[22] = 228446.17; sigma[23] = 123011.9; sigma[24] = 69246.836; sigma[25] = 178754.14; sigma[26] = 92953.38; sigma[27] = 118181.85; sigma[28] = 300837.44; sigma[29] = 60002.816; sigma[30] = 0.16780756; sigma[31] = 203541.8;
                        votes[3] = 0.31624612 - gauss(x, theta, sigma);
                        // return argmax of votes
                        uint8_t classIdx = 0;
                        float maxVotes = votes[0];

                        for (uint8_t i = 1; i < 4; i++) {
                            if (votes[i] > maxVotes) {
                                classIdx = i;
                                maxVotes = votes[i];
                            }
                        }

                        return classIdx;
                    }

                    /**
                    * Predict readable class name
                    */
                    const char* predictLabel(float *x) {
                        return idxToLabel(predict(x));
                    }

                    /**
                    * Convert class idx to readable name
                    */
                    const char* idxToLabel(uint8_t classIdx) {
                        switch (classIdx) {
                            case 0:
                            return "CIRCULO";
                            case 1:
                            return "DEFAULT";
                            case 2:
                            return "LADO";
                            case 3:
                            return "QUIETO";
                            default:
                            return "Houston we have a problem";
                        }
                    }

                protected:
                    /**
                    * Compute gaussian value
                    */
                    float gauss(float *x, float *theta, float *sigma) {
                        float gauss = 0.0f;

                        for (uint16_t i = 0; i < 32; i++) {
                            gauss += log(sigma[i]);
                            gauss += abs(x[i] - theta[i]) / sigma[i];
                        }

                        return gauss;
                    }
                };
            }
        }
    }