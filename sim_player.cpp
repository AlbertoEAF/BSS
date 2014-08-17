/** @file simple_client.c
 *
 * @brief This simple client demonstrates the most basic features of JACK
 * as they would be used by many applications.
 */

#include <iostream>
#include <limits>
#include <string>

#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>



#include "color_codes.h"

#include <jack/jack.h>

#include "custom_assert.h"

#include <iostream>

#include "Buffer.h"

#include <jack/jack.h>
//#include <jack/control.h> // server control
#include <jack/ringbuffer.h>

#include "wav.h"

#include "libs/config_parser.h"

typedef jack_default_audio_sample_t real;

typedef struct Stereo_block_data
{
  real *L;
  real *R;

  size_t samples;

} Stereo_block_data;


// Output ports
jack_port_t *o0;
jack_port_t *o1;

jack_client_t *client;



int process (jack_nframes_t nframes, void *arg)
{
  jack_default_audio_sample_t *out0, *out1;

  // Stereo data (L and R waves)
  Stereo_block_data * waves = (Stereo_block_data*)arg;

  out0 = (jack_default_audio_sample_t*) jack_port_get_buffer (o0, nframes);
  out1 = (jack_default_audio_sample_t*) jack_port_get_buffer (o1, nframes);

  // We start at the beginning but then we loop the output over and over
  static size_t data_pos = 0;


  size_t i;
  for (i = 0; i < nframes; ++i)
    {
      if (data_pos < waves->samples)
	{
	  out0[i] = waves->L[data_pos];
	  out1[i] = waves->L[data_pos];
	  ++data_pos;
	}
      else
	{
	  // Go back to the beginning and request a copy in the next cycle.
	  data_pos = 0;
	  --i; // Request the copy that didn't happen now (to avoid duplicating copying code).
	  puts(GREEN "Looping over." NOCOLOR);
	}
    }

  return 0;      
}

/**
 * JACK calls this shutdown_callback if the server ever shuts down or
 * decides to disconnect the client.
 */
void jack_shutdown (void *arg)
{
  puts(RED "JACK closed the app." NOCOLOR);
  exit (1);
}

int xrun_handler(void *arg)
{
  puts("XRun occurred!");
  //  exit(1);
  return 1;
}

/// Callback to handle changing sample rate.
int srate_changed(jack_nframes_t sample_rate_Hz, void *arg)
{
  printf(YELLOW "Changing sample rate to %u Hz.\n" NOCOLOR, sample_rate_Hz);
  return sample_rate_Hz;
}

/// Creates a JACK server at the specified sampling rate by using plughw:0.
void create_jack_server(unsigned int sample_rate)
{
  char jackd_cmd[100];
  snprintf(jackd_cmd,100, "jackd -r -dalsa -dplughw:0 -r%u -p256 -n2 &", sample_rate);
  system(jackd_cmd);


  /*
  jackctl_server_t *server;

  server = jackctl_server_create(NULL,NULL);

  const JSList *drivers = jackctl_server_get_drivers_list(server);

  Guarantee(drivers, "No drivers available.");

  int i=0;
  puts("Drivers:");
  while(drivers[i])
    {
      puts(drivers[i]);
    }
  exit(1);
  */
}

void init_jack_client(jack_client_t *client, Stereo_block_data *stereo_block_data, unsigned int sample_rate)
{
  const char **ports;
  const char *client_name = "sim_player";
  const char *server_name = NULL;
  jack_options_t options = JackNullOption;
  jack_status_t status;

  // Create a server
  
  create_jack_server(sample_rate); 

  /* open a client connection to the JACK server */

  client = jack_client_open (client_name, options, &status, server_name);
  if (client == NULL) 
    {
      fprintf (stderr, "jack_client_open() failed, status = 0x%2.0x\n", status);
      if (status & JackServerFailed)
	fprintf (stderr, "Unable to connect to JACK server\n");

      exit (1);
    }
  if (status & JackServerStarted)
    fprintf (stderr, "JACK server started\n");

  if (status & JackNameNotUnique) 
    {
      client_name = jack_get_client_name(client);
      fprintf (stderr, "unique name `%s' assigned\n", client_name);
    }


  // Register JACK callbacks
  Guarantee0 (jack_set_process_callback (client, process, stereo_block_data), "Couldn't set process callback!");

  Guarantee0 (jack_set_sample_rate_callback (client, srate_changed, 0), "Couldn't set sample rate callback!");
  Guarantee0 (jack_set_xrun_callback        (client, xrun_handler , 0), "Could not set the xrun handler");

  //  Guarantee0 (jack_set_buffer_size_callback (client, buffer_size_changed, 0), "Couldn't register buffer size callback.");

  jack_on_shutdown (client, jack_shutdown, 0);
  

  //printf ("engine sample rate: %u Hz\n", jack_get_sample_rate (client));

  /* create the ports */

  o0 = jack_port_register (client, "o0 (L)" , JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput , 0);
  o1 = jack_port_register (client, "o1 (R)" , JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput , 0);

  Guarantee (o0 && o1 , "Failed input port registration.");




  // App ready, tell JACK to activate it in the graph

  Guarantee0 (jack_activate (client), "Couldn't activate the client.");

  // Automatically connect the outputs to the L and R system outputs.
  ports = jack_get_ports (client, NULL, NULL, JackPortIsPhysical|JackPortIsInput);
  Guarantee (ports && ports[0] && ports[1], "Not enough output ports!");
  Guarantee0(jack_connect (client, jack_port_name (o0), ports[0]), "Couldn't connect to L output.");
  Guarantee0(jack_connect (client, jack_port_name (o1), ports[1]), "Couldn't connect to R output.");
  free (ports);
}

int main (int argc, char *argv[])
{
  /// Load the .wav files for the loop simulation playback

  Options o("settings.cfg", Quit, 1);

  // Choose mic input files
  std::string x1_filepath = (argc == 3 ? argv[1] : o("x1_wav"));
  std::string x2_filepath = (argc == 3 ? argv[2] : o("x2_wav"));

  SndfileHandle x1_file(x1_filepath), x2_file(x2_filepath);
  if ( ! ok(x1_file) || ! ok(x2_file) )
    return EXIT_FAILURE;

  const unsigned int sample_rate_Hz = x1_file.samplerate();
  const size_t       samples        = x1_file.frames(); 
  
  Buffer<real> x1_wav(samples), x2_wav(samples);
  x1_file.read(x1_wav(), samples);
  x2_file.read(x2_wav(), samples);

  Stereo_block_data stereo_block_data;

  stereo_block_data.L = x1_wav();
  stereo_block_data.R = x2_wav();
  stereo_block_data.samples = samples;


  printf("\nProcessing input file with %lu frames @ %u Hz.\n\n", 
	 samples, sample_rate_Hz);	

  init_jack_client(client, &stereo_block_data, sample_rate_Hz);

  puts(GREEN "Press ENTER to exit simulator." NOCOLOR);
  getchar();
  

  // Clean JACK exit
  jack_client_close (client);
  exit (0);
}
