% Moyer, Ethan 20200403
% This function accepts n, the length of the set; P, the proportions of the
% set; seq_loc, the location of the sequence data; and match, and the 
% degree of a match (from 0 exclusive to 1.0 inclusive).
% P is structured as a vector where P(1) = pa, P(2) = pt, P(3) = pc, and
% P(4) = pg.
function gatherData()
match = 0.1;
set_file = 'dna.txt';

% Searched '(mycobacterium) AND "Mycolicibacterium
% rufum"[porgn:__txid318424]' between 1000 and 5000 nucleotides under
% Batceria, Nucleotide on GenBank
warning('off', 'bioinfo:localalign:EmptyAlignment')

fragment_range = 5:20;
% for 1e4
set = lower(fileread(set_file));

for fragment_length = fragment_range
    fragments = getFragments(set, fragment_length);
    score = localalign(fragments{1}, fragments{1}, 'ALPHABET', 'nt').Score;
    for i = 1:numel(fragments)
        disp(i)
        generateNonopTable(fragments, fragments{i}, ...
            "set", fragment_length, score);
    end

end


end

function fragments = getFragments(sequence, fragment_length)
    fragments = cell(numel(sequence) - fragment_length + 1, 1);
    for i = 1:numel(sequence) - fragment_length + 1
       fragments{i} = sequence(i:i + fragment_length - 1);
    end
end

