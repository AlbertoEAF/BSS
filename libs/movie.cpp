#include "movie.h"

using namespace std;

template <class T>
int print_to_gnuplot(Matrix<T> m, std::string filepath)
{
  int i, j;
  std::ofstream out;

  out.open(filepath.c_str(), ios::trunc);


  for (i = 0; i < m.rows(); ++i)
    {
      for (j = 0; j < m.cols(); ++j)
      {
	out << i << " " << j << " " << m(i,j) << endl;

      }

    out << endl;
    }
  out.close();

  return 1; // success
}


template <class T>
int render_frame(Matrix<T> m, const string &cfg_filepath, unsigned int frame_number, unsigned int max_digits) // max_digits==4
{
  /*
  if (frame_number==0)
    {
      system("rm -rf tmp/*");
      system("mkdir -p tmp");
    }
  */

  print_to_gnuplot(m, "frame.dat");

  system((string("gnuplot ")+cfg_filepath).c_str());

  string cmd = "mv frame.png tmp/" + itosNdigits (frame_number, max_digits) + ".png";                      

  system(cmd.c_str());

}

int render_movie(unsigned int max_digits, int ffmpeg_deprecated_mode)
{
  string cmd;
  if (! ffmpeg_deprecated_mode)
    cmd = "yes | ffmpeg  -r 30 -q:v 0 -i tmp/%0" + itos(max_digits) + "d.png V_movie.mp4";
  else // para sistemas com pacotes desactualizados como o Ubuntu
    cmd = "yes | ffmpeg  -r 30 -sameq -i tmp/%0" + itos(max_digits) + "d.png V_movie.mp4";

  return system (cmd.c_str());
}




