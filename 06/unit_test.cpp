/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */


#include "Dip6.h"
#include "TaskScheduler.h"

#include <iostream>

using namespace dip6;

template<class Forward, class Backward>
bool testDataDerivative(unsigned width, unsigned height, unsigned channels, unsigned instances, std::mt19937 &rng, Forward forward, Backward backward)
{
    std::normal_distribution<float> normalDist;

    Tensor input;
    input.allocate(height, width, channels, instances);
    for (unsigned i = 0; i < input.getTotalSize(); i++)
        input[i] = normalDist(rng);
    const Tensor &output = forward(input);
    
    Tensor randomGrads;
    randomGrads.allocateLike(output);
    for (unsigned i = 0; i < randomGrads.getTotalSize(); i++)
        randomGrads[i] = normalDist(rng);
    
    const Tensor &estimatedGradients = backward(input, randomGrads);
    
    Tensor inputPos;
    Tensor inputNeg;
    inputPos.allocateLike(input);
    inputNeg.allocateLike(input);
    for (unsigned i = 0; i < input.getTotalSize(); i++)
        inputPos[i] = inputNeg[i] = input[i];
    
    float eps = 1e-3f;
    for (unsigned i = 0; i < input.getTotalSize(); i++) {
        inputPos[i] = input[i] + eps;
        inputNeg[i] = input[i] - eps;
        
        const Tensor &outputPos = forward(inputPos);
        double lossPos = 0.0;
        for (unsigned j = 0; j < randomGrads.getTotalSize(); j++)
            lossPos += randomGrads[j] * outputPos[j];
        
        const Tensor &outputNeg = forward(inputNeg);
        double lossNeg = 0.0;
        for (unsigned j = 0; j < randomGrads.getTotalSize(); j++)
            lossNeg += randomGrads[j] * outputNeg[j];
        
        float grad = (lossPos - lossNeg) / (2.0f * eps);
        float estGrad = estimatedGradients[i];
        
        if (std::abs(grad-estGrad) > 1e-1f) {
            std::cout << "Error in element " << i << std::endl;
            std::cout << "   Got: " << grad << " but expected: " << estGrad << std::endl;
            return false;
        }
        
        inputPos[i] = inputNeg[i] = input[i];
    }
    return true;
}



template<class Forward, class Backward>
bool testParamDerivative(unsigned width, unsigned height, unsigned channels, unsigned instances, std::mt19937 &rng, Tensor &parameters, Forward forward, Backward backward)
{
    std::normal_distribution<float> normalDist;
    Tensor input;
    input.allocate(height, width, channels, instances);
    for (unsigned i = 0; i < input.getTotalSize(); i++)
        input[i] = normalDist(rng);
    const Tensor &output = forward(input);
    
    Tensor randomGrads;
    randomGrads.allocateLike(output);
    for (unsigned i = 0; i < randomGrads.getTotalSize(); i++)
        randomGrads[i] = normalDist(rng);
    
    const Tensor &estimatedGradients = backward(input, randomGrads);
    
    
    float eps = 1e-3f;
    for (unsigned i = 0; i < parameters.getTotalSize(); i++) {
        float original = parameters[i];
        parameters[i] = original + eps;

        const Tensor &outputPos = forward(input);
        double lossPos = 0.0;
        for (unsigned j = 0; j < randomGrads.getTotalSize(); j++)
            lossPos += randomGrads[j] * outputPos[j];
        
        parameters[i] = original - eps;
        const Tensor &outputNeg = forward(input);
        double lossNeg = 0.0;
        for (unsigned j = 0; j < randomGrads.getTotalSize(); j++)
            lossNeg += randomGrads[j] * outputNeg[j];
        
        float grad = (lossPos - lossNeg) / (2.0f * eps);
        float estGrad = estimatedGradients[i];
        
        if (std::abs(grad-estGrad) > 1e-1f) {
            std::cout << "Error in element " << i << std::endl;
            std::cout << "   Got: " << grad << " but expected: " << estGrad << std::endl;
            return false;
        }
        
        
        parameters[i] = original;
    }
    return true;
}


bool test_MSELoss() 
{
    std::mt19937 rng(1234);
    
    dip6::MSELoss mseLoss;
    dip6::Tensor A, B, grad;
    
    std::cout << "==== test_MSELoss ====" << std::endl;
    std::cout << "    Testing random sizes to check for crashes" << std::endl;
    for (unsigned i = 0; i < 100; i++) {
        std::uniform_int_distribution<int> dist(0, 16);
        A.allocate(24+dist(rng), 24+dist(rng), 1+dist(rng), 16);
        B.allocate(A.getSize(0)-dist(rng), A.getSize(1)-dist(rng), A.getSize(2), 16);
        
        mseLoss.computeLoss(B, A, grad);
    }
    
    A.allocate(128, 64, 2, 16);
    B.allocate(4, 8, 2, 16);

    std::normal_distribution<float> normalDist;

    std::cout << "    Testing with predefined pattern" << std::endl;

    for (unsigned i = 0; i < A.getTotalSize(); i++)
        A[i] = std::cos(i*0.001f);
    for (unsigned i = 0; i < B.getTotalSize(); i++)
        B[i] = std::cos(i*0.02f);
    
    float expectedLoss = 0.883455f;
    float expectedGrad[] = {0.00311561f, 0.00311679f, 0.00311719f, 0.0031168f, 0.00311563f, 0.00311368f, 0.00311096f, 0.00310746f, 0.00310318f, 0.00309813f, 0.00309231f, 0.00308572f, 0.00307837f, 0.00307027f, 0.00306141f, 0.00305179f, 0.00304142f, 0.00303032f, 0.00301847f, 0.00300589f, 0.00299259f, 0.00297856f, 0.00296382f, 0.00294837f, 0.00293222f, 0.00291538f, 0.00289785f, 0.00287965f, 0.00286077f, 0.00284122f, 0.00282102f, 0.00280018f, 0.0027787f, 0.00275659f, 0.00273386f, 0.00271052f, 0.00268659f, 0.00266206f, 0.00263696f, 0.00261129f, 0.00258507f, 0.00255831f, 0.002531f, 0.00250317f, 0.00247483f, 0.00244599f, 0.00241667f, 0.00238687f, 0.00235661f, 0.0023259f, 0.00229475f, 0.00226318f, 0.0022312f, 0.00219882f, 0.00216606f, 0.00213294f, 0.00209945f, 0.00206563f, 0.00203147f, 0.001997f, 0.00196224f, 0.00192718f, 0.00189186f, 0.00185629f, 0.00182047f, 0.00178443f, 0.00174818f, 0.00171173f, 0.0016751f, 0.00163832f, 0.00160138f, 0.00156431f, 0.00152711f, 0.00148982f, 0.00145243f, 0.00141498f, 0.00137746f, 0.00133991f, 0.00130233f, 0.00126474f, 0.00122716f, 0.0011896f, 0.00115207f, 0.00111461f, 0.00107721f, 0.00103989f, 0.00100267f, 0.000965561f, 0.000928586f, 0.000891756f, 0.000855088f, 0.000818595f, 0.000782294f, 0.0007462f, 0.000710327f, 0.00067469f, 0.000639305f, 0.000604196f, 0.000569357f, 0.000534814f, 0.00050058f, 0.000466669f, 0.000433097f, 0.000399876f, 0.000367022f, 0.000334547f, 0.000302464f, 0.000270789f, 0.000239533f, 0.00020871f, 0.000178343f, 0.000148424f, 0.000118976f, 9.00116e-05f, 6.15424e-05f, 3.35806e-05f, 6.138e-06f, -2.07738e-05f, -4.71437e-05f, -7.29603e-05f, -9.8213e-05f, -0.000122891f, -0.000146984f, -0.000170481f, -0.000193362f, -0.000215639f, -0.000237292f, -0.00025831f, -0.000278697f, -0.000298411f, -0.000317486f, -0.000335872f, -0.000353582f, -0.00037063f, -0.000386965f, -0.000402624f, -0.000417556f, -0.000431798f, -0.000445301f, -0.000458102f, -0.000470152f, -0.000481489f, -0.000492065f, -0.000501917f, -0.000510999f, -0.000519328f, -0.00052692f, -0.000533731f, -0.000539799f, -0.000545079f, -0.00054961f, -0.000553349f, -0.000556334f, -0.000558522f, -0.000559954f, -0.000560587f, -0.000560462f, -0.000559536f, -0.000557851f, -0.000555365f, -0.0005521f, -0.000548078f, -0.000543257f, -0.000537681f, -0.00053131f, -0.000524188f, -0.000516274f, -0.000507614f, -0.000498169f, -0.000487983f, -0.000477019f, -0.000465321f, -0.000452854f, -0.00043964f, -0.000425707f, -0.000411018f, -0.000395619f, -0.000379476f, -0.000362634f, -0.000345059f, -0.000326798f, -0.000307819f, -0.000288167f, -0.00026781f, -0.000246797f, -0.000225094f, -0.00020275f, -0.000179733f, -0.000156072f, -0.000131796f, -0.000106874f, -8.1356e-05f, -5.52108e-05f, -2.84889e-05f, -1.16008e-06f, 2.67248e-05f, 5.51954e-05f, 8.4201e-05f, 0.00011377f, 0.000143852f, 0.000174475f, 0.000205606f, 0.000237215f, 0.00026933f, 0.000301896f, 0.000334944f, 0.000368419f, 0.000402348f, 0.00043668f, 0.000471439f, 0.000506574f, 0.00054211f, 0.000577995f, 0.000614253f, 0.000650831f, 0.000687755f, 0.000724991f, 0.000762504f, 0.00080032f, 0.000838385f, 0.000876722f, 0.000915278f, 0.000954078f, 0.000993066f, 0.00103227f, 0.00107163f, 0.00111117f, 0.00115084f, 0.00119067f, 0.00123061f, 0.00127063f, 0.00131076f, 0.00135094f, 0.00139119f, 0.00143147f, 0.00147179f, 0.00151209f, 0.00155241f, 0.00159269f, 0.00163294f, 0.00167313f, 0.00171326f, 0.0017533f, 0.00179323f, 0.00183306f, 0.00187274f, 0.00191229f, 0.00195166f, 0.00199087f, 0.00202987f, 0.00206868f, 0.00210724f, 0.00214559f, 0.00218367f, 0.00222149f, 0.00225902f, 0.00163447f, 0.00166844f, 0.00170205f, 0.00173535f, 0.00176828f, 0.00180086f, 0.00183303f, 0.00186484f, 0.00189621f, 0.00192719f, 0.00195771f, 0.00198781f, 0.00201742f, 0.00204658f, 0.00207524f, 0.0021034f, 0.00213107f, 0.0021582f, 0.00218482f, 0.00221088f, 0.0022364f, 0.00226134f, 0.00228572f, 0.00230949f, 0.0023327f, 0.00235526f, 0.00237724f, 0.00239857f, 0.00241926f, 0.00243933f, 0.00245872f, 0.00247748f, 0.00249554f, 0.00251295f, 0.00252965f, 0.00254569f, 0.00256099f, 0.00257562f, 0.00258951f, 0.00260271f, 0.00261515f, 0.00262686f, 0.00263786f, 0.0026481f, 0.00265761f, 0.00266635f, 0.00267435f, 0.00268158f, 0.00268806f, 0.00269376f, 0.00269871f, 0.00270286f, 0.00270627f, 0.00270887f, 0.00271072f, 0.00271177f, 0.00271203f, 0.00271155f, 0.00271025f, 0.0027082f, 0.00270533f, 0.00270172f, 0.00269729f, 0.00269212f, 0.00268614f, 0.00267941f, 0.00267187f, 0.0026636f, 0.00265452f, 0.00264468f, 0.0026341f, 0.00262273f, 0.00261064f, 0.00259776f, 0.00258417f, 0.0025698f, 0.00255473f, 0.00253889f, 0.00252236f, 0.00250507f, 0.0024871f, 0.00246839f, 0.00244901f, 0.0024289f, 0.00240811f, 0.00238667f, 0.00236452f, 0.00234174f, 0.00231828f, 0.0022942f, 0.00226945f, 0.0022441f, 0.0022181f, 0.00219152f, 0.00216432f, 0.00213655f, 0.00210817f, 0.00207923f, 0.00204975f, 0.00201971f, 0.00198915f, 0.00195804f, 0.00192645f, 0.00189433f, 0.00186175f, 0.00182866f, 0.00179514f, 0.00176114f, 0.00172673f, 0.00169187f, 0.00165663f, 0.00162096f, 0.0015849f, 0.0015485f, 0.00151171f, 0.00147461f, 0.00143716f, 0.00139941f, 0.00136134f, 0.00132301f, 0.00128438f, 0.00124552f, 0.00120639f, 0.00116706f, 0.0011275f, 0.00108773f, 0.00104781f, 0.0010077f, 0.000967465f, 0.000927064f, 0.000886572f, 0.000845946f, 0.00080526f, 0.000764471f, 0.000723651f, 0.000682761f, 0.000641872f, 0.000600943f, 0.000560047f, 0.000519142f, 0.000478272f, 0.000437481f, 0.000396728f, 0.000356087f, 0.000315514f, 0.000275084f, 0.000234752f, 0.000194594f, 0.000154565f, 0.00011474f, 7.50757e-05f, 3.5646e-05f, -3.59421e-06f, -4.25978e-05f, -8.13252e-05f, -0.000119817f, -0.000157999f, -0.000195917f, -0.000233498f, -0.000270787f, -0.000307709f, -0.00034431f, -0.000380517f, -0.000416377f, -0.000451814f, -0.000486876f, -0.000521519f, -0.000555703f, -0.000589469f, -0.000622752f, -0.000655591f, -0.000687922f, -0.000719785f, -0.000751114f, -0.000781953f, -0.000812235f, -0.000842003f, -0.000871192f, -0.000899845f, -0.000927896f, -0.000955392f, -0.000982292f, -0.00100856f, -0.00103424f, -0.00105927f, -0.0010837f, -0.00110745f, -0.00113058f, -0.00115303f, -0.00117483f, -0.00119593f, -0.00121638f, -0.0012361f, -0.00125516f, -0.00127351f, -0.00129113f, -0.00130805f, -0.00132422f, -0.00133969f, -0.0013544f, -0.0013684f, -0.00138162f, -0.00139412f, -0.00140584f, -0.00141683f, -0.00142703f, -0.0014365f, -0.00144517f, -0.0014531f, -0.00146025f, -0.00146661f, -0.00147221f, -0.00147701f, -0.00148106f, -0.00148429f, -0.00148678f, -0.00148846f, -0.00148938f, -0.0014895f, -0.00148887f, -0.00148742f, -0.00148523f, -0.00148227f, -0.0014785f, -0.00147399f, -0.00146868f, -0.00146264f, -0.0014558f, -0.00144823f, -0.00143988f, -0.00143081f, -0.00142096f, -0.00141039f, -0.00139906f, -0.00138703f, -0.00137424f, -0.00136076f, -0.00134657f, -0.00133164f, -0.00131603f, -0.00129971f, -0.00128272f, -0.00126503f, -0.00124669f, -0.00122766f, -0.00120801f, -0.00118768f, -0.00116674f, -0.00114515f, -0.00112296f, -0.00110017f, -0.00107674f, -0.00105276f, -0.00102817f, -0.00100304f, -0.000977319f, -0.00329196f, -0.00326322f, -0.00323395f, -0.00320417f, -0.00317389f, -0.00314311f, -0.00311186f, -0.00308014f, -0.00304797f, -0.00301536f, -0.00298232f, -0.00294887f, -0.00291501f, -0.00288077f, -0.00284616f, -0.00281118f, -0.00277586f, -0.00274021f, -0.00270425f, -0.00266797f, -0.00263141f, -0.00259458f, -0.00255749f, -0.00252015f, -0.00248258f, -0.0024448f, -0.00240682f, -0.00236865f, -0.00233032f, -0.00229183f, -0.00225321f, -0.00221446f, -0.0021756f, -0.00213665f, -0.00209763f, -0.00205854f, -0.00201941f, -0.00198025f, -0.00194107f, -0.0019019f, -0.00186274f, -0.00182362f, -0.00178454f, -0.00174553f, -0.00170659f, -0.00166776f, -0.00162903f, -0.00159043f, -0.00155197f, -0.00151367f, -0.00147553f, -0.00143759f, -0.00139985f, -0.00136232f, -0.00132503f, -0.00128798f, -0.0012512f, -0.00121469f, -0.00117848f, -0.00114257f, -0.00110698f, -0.00107172f, -0.00103681f, -0.00100227f, -0.000968093f, -0.000934312f, -0.000900932f, -0.000867969f, -0.000835434f, -0.000803342f, -0.000771704f, -0.000740535f, -0.000709844f, -0.000679646f, -0.000649951f, -0.000620773f, -0.000592121f, -0.000564006f, -0.000536445f, -0.000509442f, -0.000483013f, -0.000457164f, -0.000431909f, -0.000407255f, -0.000383215f, -0.000359794f, -0.000337007f, -0.000314857f, -0.000293359f, -0.000272515f, -0.00025234f, -0.000232835f, -0.000214011f, -0.000195879f, -0.00017844f, -0.000161707f, -0.00014568f, -0.000130372f, -0.000115782f, -0.000101923f, -8.87933e-05f, -7.64058e-05f, -6.47568e-05f, -5.38591e-05f, -4.37099e-05f, -3.4316e-05f, -2.56854e-05f, -1.78135e-05f, -1.07116e-05f, -4.37396e-06f, 1.18872e-06f, 5.98235e-06f, 9.99693e-06f, 1.32393e-05f, 1.57e-05f, 1.73864e-05f, 1.82897e-05f, 1.84177e-05f, 1.77666e-05f, 1.63325e-05f, 1.41244e-05f, 1.11349e-05f, 7.37386e-06f, 2.83401e-06f, -2.47348e-06f, -8.55559e-06f, -1.54008e-05f, -2.30153e-05f, -3.13866e-05f, -4.05207e-05f, -5.04048e-05f, -6.10432e-05f, -7.24234e-05f, -8.45441e-05f, -9.74063e-05f, -0.000110994f, -0.000125313f, -0.000140345f, -0.000156097f, -0.00017255f, -0.000189709f, -0.000207556f, -0.000226094f, -0.000245305f, -0.000265193f, -0.000285737f, -0.000306936f, -0.000328786f, -0.000351268f, -0.000374381f, -0.000398109f, -0.000422449f, -0.000447385f, -0.000472913f, -0.000499015f, -0.00052569f, -0.000552916f, -0.000580694f, -0.000609001f, -0.000637837f, -0.000667178f, -0.00069702f, -0.000727353f, -0.000758158f, -0.00078943f, -0.000821148f, -0.000853307f, -0.000885887f, -0.000918882f, -0.00095227f, -0.000986047f, -0.00102019f, -0.0010547f, -0.00108954f, -0.00112471f, -0.0011602f, -0.00119599f, -0.00123207f, -0.00126841f, -0.00130502f, -0.00134186f, -0.00137893f, -0.00141621f, -0.00145369f, -0.00149135f, -0.00152918f, -0.00156716f, -0.00160528f, -0.00164351f, -0.00168185f, -0.00172028f, -0.00175878f, -0.00179735f, -0.00183595f, -0.00187458f, -0.00191322f, -0.00195187f, -0.00199048f, -0.00202907f, -0.0020676f, -0.00210607f, -0.00214445f, -0.00218273f, -0.00222091f, -0.00225895f, -0.00229686f, -0.00233459f, -0.00237217f, -0.00240954f, -0.00244672f, -0.00248366f, -0.00252039f, -0.00255685f, -0.00259306f, -0.00262898f, -0.00266461f, -0.00269993f, -0.00273493f, -0.0027696f, -0.00280391f, -0.00283786f, -0.00287142f, -0.0029046f, -0.00293737f, -0.00296973f, -0.00300165f, -0.00303313f, -0.00306414f, -0.0030947f, -0.00312476f, -0.00315433f, -0.0031834f, -0.00321194f, -0.00323996f, -0.00326743f, -0.00329435f, -0.00332071f, -0.00334649f, -0.00337169f, -0.00339629f, -0.00342029f, -0.00344367f, -0.00346643f, -0.00348854f, -0.00351003f, -0.00353085f, -0.00355102f, -0.00357052f, -0.00358934f, -0.00360748f, -0.00362493f, -0.00364167f, -0.00365771f, -0.00367304f, -0.00368765f, -0.00370153f, -0.000902577f, -0.000913802f, -0.000924314f, -0.000934056f, -0.000943076f, -0.000951318f, -0.000958829f, -0.000965554f, -0.000971542f, -0.000976737f, -0.000981189f, -0.000984844f, -0.00098775f, -0.000989855f, -0.000991208f, -0.000991757f, -0.000991527f, -0.000990542f, -0.000988752f, -0.000986207f, -0.000982856f, -0.000978752f, -0.000973844f, -0.000968185f, -0.000961725f, -0.000954517f, -0.000946512f, -0.000937764f, -0.000928224f, -0.000917922f, -0.000906886f, -0.000895069f, -0.000882526f, -0.00086921f, -0.000855178f, -0.000840382f, -0.000824879f, -0.000808624f, -0.000791674f, -0.000773983f, -0.000755609f, -0.000736508f, -0.000716714f, -0.000696255f, -0.000675092f, -0.000653281f, -0.000630783f, -0.000607653f, -0.000583849f, -0.000559433f, -0.000534364f, -0.000508699f, -0.000482398f, -0.000455521f, -0.000428029f, -0.000399985f, -0.000371343f, -0.000342143f, -0.00031242f, -0.000282138f, -0.000251354f, -0.000220031f, -0.000188232f, -0.000155921f, -0.000123155f, -8.98988e-05f, -5.62161e-05f, -2.2072e-05f, 1.24763e-05f, 4.7463e-05f, 8.28493e-05f, 0.000118594f, 0.000154739f, 0.000191218f, 0.000228067f, 0.000265218f, 0.000302715f, 0.000340489f, 0.000378575f, 0.00041691f, 0.000455525f, 0.000494362f, 0.000533453f, 0.000572733f, 0.000612233f, 0.00065192f, 0.000691751f, 0.00073176f, 0.000771878f, 0.000812147f, 0.000852499f, 0.000892966f, 0.000933481f, 0.000974085f, 0.00101471f, 0.00105539f, 0.00109605f, 0.00113674f, 0.00117741f, 0.00121803f, 0.00125861f, 0.00129911f, 0.00133956f, 0.00137988f, 0.00142012f, 0.0014602f, 0.00150017f, 0.00153996f, 0.0015796f, 0.00161902f, 0.00165827f, 0.00169728f, 0.00173608f, 0.00177463f, 0.00181289f, 0.0018509f, 0.00188859f, 0.001926f, 0.00196307f, 0.00199983f, 0.00203621f, 0.00207226f, 0.0021079f, 0.00214318f, 0.00217803f, 0.00221249f, 0.00224651f, 0.00228008f, 0.00231321f, 0.00234584f, 0.00237802f, 0.00240968f, 0.00244086f, 0.0024715f, 0.00250164f, 0.00253121f, 0.00256025f, 0.00258871f, 0.00261662f, 0.00264392f, 0.00267065f, 0.00269678f, 0.00272227f, 0.00274716f, 0.00277139f, 0.00279501f, 0.00281795f, 0.00284026f, 0.00286187f, 0.00288284f, 0.00290309f, 0.00292268f, 0.00294155f, 0.00295974f, 0.00297722f, 0.00299395f, 0.00300998f, 0.00302526f, 0.00303983f, 0.00305364f, 0.00306672f, 0.00307903f, 0.00309061f, 0.00310141f, 0.00311147f, 0.00312074f, 0.00312926f, 0.00313702f, 0.00314397f, 0.00315018f, 0.00315558f, 0.00316023f, 0.00316407f, 0.00316716f, 0.00316944f, 0.00317096f, 0.00317168f, 0.00317163f, 0.00317078f, 0.00316918f, 0.00316677f, 0.00316361f, 0.00315967f, 0.00315493f, 0.00314945f, 0.00314318f, 0.00313616f, 0.00312836f, 0.00311982f, 0.00311051f, 0.00310047f, 0.00308966f, 0.00307813f, 0.00306585f, 0.00305286f, 0.00303915f, 0.00302469f, 0.00300955f, 0.00299368f, 0.00297713f, 0.00295988f, 0.00294196f, 0.00292334f, 0.00290408f, 0.00288414f, 0.00286357f, 0.00284234f, 0.00282049f, 0.002798f, 0.00277492f, 0.00275124f, 0.00272694f, 0.00270208f, 0.00267663f, 0.00265063f, 0.00262406f, 0.00259698f, 0.00256935f, 0.00254121f, 0.00251256f, 0.00248343f, 0.0024538f, 0.00242372f, 0.00239319f, 0.00236219f, 0.00233079f, 0.00229895f, 0.00226673f, 0.00223409f, 0.0022011f, 0.00216773f, 0.00213403f, 0.00209997f, 0.00206562f, 0.00203094f, 0.00199598f, 0.00196073f, 0.00192524f, 0.00188951f, 0.00185351f, 0.00181733f, 0.00178091f, 0.00174434f, 0.00170757f, 0.00167066f, 0.00163359f, 0.00159642f, 0.00155911f, 0.00152173f, 0.00148425f, 0.00144673f, 0.00140916f, 0.00137153f, 0.00133391f, 0.00129626f, 0.00125865f, };

    float loss = mseLoss.computeLoss(B, A, grad);
    
    if (std::abs(loss - expectedLoss) > 0.01f) {
        std::cout << "Error: Expected a loss of " << expectedLoss << " but got " << loss << std::endl;
        return false;
    }
    
    if ((grad.getSize(0) != B.getSize(0)) ||
        (grad.getSize(1) != B.getSize(1)) ||
        (grad.getSize(2) != B.getSize(2)) ||
        (grad.getSize(3) != B.getSize(3))) {
        std::cout << "Error: Grad must have the same size as the computed output" << std::endl;
        return false;
    }
        
    for (unsigned i = 0; i < grad.getTotalSize(); i++)
        if (std::abs(grad[i] - expectedGrad[i]) > 0.0001f) {
            std::cout << "Error in gradient element " << i << ": Expected a grad of " << expectedGrad[i] << " but got " << grad[i] << std::endl;
            return false;
        }
    
    return true;
}


int main(int argc, char** argv) {
    TaskScheduler::Init(std::thread::hardware_concurrency());
    
    bool ok = true;
    
    std::mt19937 rng;
    
    layers::ConvReference convLayer(1, 1, 3, 16);
    convLayer.initialize(rng);
    
    std::cout << "Testing reference conv data derivative" << std::endl;
    ok &= testDataDerivative(8, 4, 3, 16, rng, 
        [&](Tensor &input)->const Tensor&{
            convLayer.forward(input);
            return convLayer.getLastOutput();
        },
        [&](Tensor &input, Tensor &grad)->const Tensor&{
            convLayer.backward(input, grad);
            return convLayer.getLastInputGradients();
        }
    );  
    
    std::cout << "Testing reference conv kernel derivative" << std::endl;
    ok &= testParamDerivative(8, 4, 3, 16, rng, convLayer.getParameters()[0],
        [&](Tensor &input)->const Tensor&{
            convLayer.forward(input);
            return convLayer.getLastOutput();
        },
        [&](Tensor &input, Tensor &grad)->const Tensor&{
            convLayer.backward(input, grad);
            return convLayer.getParameterGradients()[0];
        }
    );  
    std::cout << "Testing reference conv bias derivative" << std::endl;
    ok &= testParamDerivative(8, 4, 3, 16, rng, convLayer.getParameters()[1],
        [&](Tensor &input)->const Tensor&{
            convLayer.forward(input);
            return convLayer.getLastOutput();
        },
        [&](Tensor &input, Tensor &grad)->const Tensor&{
            convLayer.backward(input, grad);
            return convLayer.getParameterGradients()[1];
        }
    );  
    
    layers::ConvOptimized convLayer2(1, 1, 3, 16);
    convLayer2.initialize(rng);
    
    std::cout << "Testing optimized conv data derivative" << std::endl;
    ok &= testDataDerivative(8, 4, 3, 16, rng, 
        [&](Tensor &input)->const Tensor&{
            convLayer2.forward(input);
            return convLayer2.getLastOutput();
        },
        [&](Tensor &input, Tensor &grad)->const Tensor&{
            convLayer2.backward(input, grad);
            return convLayer2.getLastInputGradients();
        }
    );  
    
    std::cout << "Testing optimized conv kernel derivative" << std::endl;
    ok &= testParamDerivative(8, 4, 3, 16, rng, convLayer2.getParameters()[0],
        [&](Tensor &input)->const Tensor&{
            convLayer2.forward(input);
            return convLayer2.getLastOutput();
        },
        [&](Tensor &input, Tensor &grad)->const Tensor&{
            convLayer2.backward(input, grad);
            return convLayer2.getParameterGradients()[0];
        }
    );  
    std::cout << "Testing optimized conv bias derivative" << std::endl;
    ok &= testParamDerivative(8, 4, 3, 16, rng, convLayer2.getParameters()[1],
        [&](Tensor &input)->const Tensor&{
            convLayer2.forward(input);
            return convLayer2.getLastOutput();
        },
        [&](Tensor &input, Tensor &grad)->const Tensor&{
            convLayer2.backward(input, grad);
            return convLayer2.getParameterGradients()[1];
        }
    );  
    
    
    layers::ReLU reluLayer;
    std::cout << "Testing ReLU data derivative" << std::endl;
    ok &= testDataDerivative(8, 4, 3, 16, rng, 
        [&](Tensor &input)->const Tensor&{
            reluLayer.forward(input);
            return reluLayer.getLastOutput();
        },
        [&](Tensor &input, Tensor &grad)->const Tensor&{
            reluLayer.backward(input, grad);
            return reluLayer.getLastInputGradients();
        }
    );  
    
    layers::Upsample upsampleLayer(3, 3);
    std::cout << "Testing Upsample data derivative" << std::endl;
    ok &= testDataDerivative(8, 4, 3, 16, rng, 
        [&](Tensor &input)->const Tensor&{
            upsampleLayer.forward(input);
            return upsampleLayer.getLastOutput();
        },
        [&](Tensor &input, Tensor &grad)->const Tensor&{
            upsampleLayer.backward(input, grad);
            return upsampleLayer.getLastInputGradients();
        }
    );  
    
    ok &= test_MSELoss();
    
    if (ok) {
        std::cout << "Unit tests seem ok" << std::endl;
        return 0;
    } else {
        std::cout << "There are still errors" << std::endl;
        return -1;
    }
} 
