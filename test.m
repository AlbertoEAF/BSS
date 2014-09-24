function [Sefiles,Sfiles]=test(Sepattern, Spattern, x0)


Sefiles = dir(Sepattern)
Sfiles  = dir( Spattern)

display(size(Sefiles));
display(Sfiles);

end
