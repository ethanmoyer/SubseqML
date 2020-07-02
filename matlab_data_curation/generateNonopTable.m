% Moyer, Ethan 20200329
% Given the location of a randomly generated set of DNA or the location of 
% a prebuilt set of DNA, this function generates
% nonoptimized data based on specific condiitons given, invluding the buffer
% length, the sequence under question, and the degree of a match (from 0 
% exclusive to 1.0 inclusive). The function returns the data and writes it
% to data_nonop.csv.
% NOTE: Include naming convention.
% Still need to introduce the match variable.
% Can create separate function for generating non op and op data since 
% nonopdata is dependent on less variables, and thus doesn't need to be
% generated each time op is generated.
function generateNonopTable(set, sequence, name, length, score)
    % Nonoptimized
    set_size = size(set, 1);
    T1 = table();
    T1.('Subsequence')(1:set_size) = "";
    T1.('Contains')(1:set_size) = 0;
    for i = 1:set_size
        T1.('Subsequence')(i) = string(set(i, :));
        percent_score = 1 - (score - localalign(string(set(i, :)), sequence, ...
            'ALPHABET', 'nt').Score) / score;
        if isempty(percent_score)
            percent_score = 0;
        end
        T1.('Contains')(i) = percent_score;
    end
    file_name = "data1_1e3/" + name + "_" + sequence + "_" + ...
    	length + ".csv";
    writetable(T1, file_name);
end

