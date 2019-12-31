function [BW,maskedRGBImage] = createMask(RGB)
I = rgb2hsv(RGB);
channel1Min = 0.056;
channel1Max = 0.136;
channel2Min = 0.128;
channel2Max = 1.000;
channel3Min = 0.087;
channel3Max = 0.913;
sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
    (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
    (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
BW = sliderBW;
maskedRGBImage = RGB;
maskedRGBImage(repmat(~BW,[1 1 3])) = 0;
end