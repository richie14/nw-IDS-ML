import pandas as pd
import xml.etree.ElementTree as et

cols = ["appName", "totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets", "totalSourcePackets", "sourcePayloadAsBase64", "destinationPayloadAsBase64", "destinationPayloadAsUTF", "direction", "sourceTCPFlagsDescription", "destinationTCPFlagsDescription", "source", "protocolName", "sourcePort", "destination", "destinationPort","startDateTime","stopDateTime","Tag"]

parse_XML('TestbedSatJun12Flows.xml',cols)

def parse_XML(xml_file, df_cols): 
    
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    
    for node in xroot: 
        res = []
        res.append(node.attrib.get(df_cols[0]))
        for el in df_cols[1:]: 
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else: 
                res.append(None)
        rows.append({df_cols[i]: res[i] 
                     for i, _ in enumerate(df_cols)})
    
    out_df = pd.DataFrame(rows, columns=df_cols)
        
    return out_df
