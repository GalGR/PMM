function [V, A, B, C, D] = read_mat(mat_filename, precision)
    fid = fopen(mat_filename);
    mat_size_pos = ftell(fid);
    mat_size = fread(fid, 2, "uint64"); pos = ftell(fid)
    mat_V_pos = ftell(fid);
    rows = mat_size(1);
    cols = mat_size(2);
    V_len = rows * cols;
    V_arr_X = fread(fid, V_len, precision); pos = ftell(fid)
    V_arr_Y = fread(fid, V_len, precision); pos = ftell(fid)
    V_arr_Z = fread(fid, V_len, precision); pos = ftell(fid)
    V_X = reshape(V_arr_X, cols, rows).';
    V_Y = reshape(V_arr_Y, cols, rows).';
    V_Z = reshape(V_arr_Z, cols, rows).';
    V = cat(3, V_X, V_Y, V_Z);
    mat_C_pos = ftell(fid);
    C_len = (rows - 1) * (cols - 1);
    A(:, :, 1, 1) = reshape(fread(fid, C_len, precision),     [cols - 1, rows - 1]      ).'; pos = ftell(fid) % 'a' upwards left
    B(:, :, 1, 1) = reshape(fread(fid, C_len * 2, precision), [(cols - 1) * 2, rows - 1]).'; pos = ftell(fid) % 'b' upwards left
    C(:, :, 1, 1) = reshape(fread(fid, C_len * 4, precision), [(cols - 1) * 4, rows - 1]).'; pos = ftell(fid) % 'c' upwards left
    A(:, :, 2, 1) = reshape(fread(fid, C_len, precision),     [cols - 1, rows - 1]      ).'; pos = ftell(fid) % 'a' upwards right
    B(:, :, 2, 1) = reshape(fread(fid, C_len * 2, precision), [(cols - 1) * 2, rows - 1]).'; pos = ftell(fid) % 'b' upwards right
    C(:, :, 2, 1) = reshape(fread(fid, C_len * 4, precision), [(cols - 1) * 4, rows - 1]).'; pos = ftell(fid) % 'c' upwards right
    A(:, :, 1, 2) = reshape(fread(fid, C_len, precision),     [cols - 1, rows - 1]      ).'; pos = ftell(fid) % 'a' downwards left
    B(:, :, 1, 2) = reshape(fread(fid, C_len * 2, precision), [(cols - 1) * 2, rows - 1]).'; pos = ftell(fid) % 'b' downwards left
    C(:, :, 1, 2) = reshape(fread(fid, C_len * 4, precision), [(cols - 1) * 4, rows - 1]).'; pos = ftell(fid) % 'c' downwards left
    A(:, :, 2, 2) = reshape(fread(fid, C_len, precision),     [cols - 1, rows - 1]      ).'; pos = ftell(fid) % 'a' downwards right
    B(:, :, 2, 2) = reshape(fread(fid, C_len * 2, precision), [(cols - 1) * 2, rows - 1]).'; pos = ftell(fid) % 'b' downwards right
    C(:, :, 2, 2) = reshape(fread(fid, C_len * 4, precision), [(cols - 1) * 4, rows - 1]).'; pos = ftell(fid) % 'c' downwards right
    A(:, :, 1, 3) = reshape(fread(fid, C_len, precision),     [cols - 1, rows - 1]      ).'; pos = ftell(fid) % 'a' rightwards left
    B(:, :, 1, 3) = reshape(fread(fid, C_len * 2, precision), [(cols - 1) * 2, rows - 1]).'; pos = ftell(fid) % 'b' rightwards left
    C(:, :, 1, 3) = reshape(fread(fid, C_len * 4, precision), [(cols - 1) * 4, rows - 1]).'; pos = ftell(fid) % 'c' rightwards left
    A(:, :, 2, 3) = reshape(fread(fid, C_len, precision),     [cols - 1, rows - 1]      ).'; pos = ftell(fid) % 'a' rightwards right
    B(:, :, 2, 3) = reshape(fread(fid, C_len * 2, precision), [(cols - 1) * 2, rows - 1]).'; pos = ftell(fid) % 'b' rightwards right
    C(:, :, 2, 3) = reshape(fread(fid, C_len * 4, precision), [(cols - 1) * 4, rows - 1]).'; pos = ftell(fid) % 'c' rightwards right
    A(:, :, 1, 4) = reshape(fread(fid, C_len, precision),     [cols - 1, rows - 1]      ).'; pos = ftell(fid) % 'a' leftwards left
    B(:, :, 1, 4) = reshape(fread(fid, C_len * 2, precision), [(cols - 1) * 2, rows - 1]).'; pos = ftell(fid) % 'b' leftwards left
    C(:, :, 1, 4) = reshape(fread(fid, C_len * 4, precision), [(cols - 1) * 4, rows - 1]).'; pos = ftell(fid) % 'c' leftwards left
    A(:, :, 2, 4) = reshape(fread(fid, C_len, precision),     [cols - 1, rows - 1]      ).'; pos = ftell(fid) % 'a' leftwards right
    B(:, :, 2, 4) = reshape(fread(fid, C_len * 2, precision), [(cols - 1) * 2, rows - 1]).'; pos = ftell(fid) % 'b' leftwards right
    C(:, :, 2, 4) = reshape(fread(fid, C_len * 4, precision), [(cols - 1) * 4, rows - 1]).'; pos = ftell(fid) % 'c' leftwards right
    mat_D_pos = ftell(fid);
    D_arr = fread(fid, V_len, precision); pos = ftell(fid)
    D = reshape(D_arr, cols, rows).';
end