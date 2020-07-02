% Moyer, Ethan 20200329
% This function generates a dna molecule of length n with the appropriate
% proporitons pa, pt, pc, and pg. The function returns the randomly
% generated set of DNA and writes it to dna.txt.

% Implement usng randseq

function dna = generateDNA(n,pa,pt,pc,pg)
if pa + pt + pc + pg ~= 1
    disp("Please provide proportions that sum to 1.0")
end
if n < 0
    disp("Please provide a non-negative/nonzero size for the size of the molecule")
end
if pa + pc ~= pt + pg
    disp("Please provide proportions that satisfy pa + pc = pt + pg")
end
dna = '';
if pa + pt + pc + pg == 1 && n > 0 && pa + pc == pt + pg
    for i = 1:n
        ranNum = rand(1);
        if ranNum <= pa
            dna(i) = "A";
        elseif ranNum <= pt + pa
            dna(i) = "T";
        elseif ranNum <= pc + pt + pa
            dna(i) = "C";
        else
            dna(i) = "G";
        end
    end
end
writematrix(dna)
end

