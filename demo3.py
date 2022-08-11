
from flowmeter import Flowmeter

feature_gen = Flowmeter(
    offline = "output_packet_capture.pcap",
    outfunc = None,
    outfile = "output.csv")
feature_gen.run()

