import numpy as np
        
class ra_encoder():
    def __init__(self, geometry, statistics,regression_layer = 2):
        
        self.geometry = geometry
        self.statistics = statistics
        self.regression_layer = regression_layer

        self.INPUT_DIM = (geometry['ranges'][0],geometry['ranges'][1],geometry['ranges'][2])
        self.OUTPUT_DIM = (regression_layer + 2,self.INPUT_DIM[0] // 4 , self.INPUT_DIM[1] // 4 )

    def encode(self,labels):
        map = np.zeros(self.OUTPUT_DIM)

        for lab in labels:
            # [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix, class_name]

            if(lab[0]==-1):
                continue

            range_bin = int(np.clip(lab[0]/self.geometry['resolution'][0]/4,0,self.OUTPUT_DIM[1]))
            range_mod = lab[0] - range_bin*self.geometry['resolution'][0]*4

            # ANgle and deg
            angle_bin = int(np.clip(np.floor(lab[1]/self.geometry['resolution'][1]/4 + self.OUTPUT_DIM[2]/2),0,self.OUTPUT_DIM[2]))
            angle_mod = lab[1] - (angle_bin- self.OUTPUT_DIM[2]/2)*self.geometry['resolution'][1]*4 
            
            class_id = lab[10]

            if(self.geometry['size']==1):
                map[0,range_bin,angle_bin] = 1
                map[1,range_bin,angle_bin] = (range_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                map[2,range_bin,angle_bin] = (angle_mod - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
                map[3, range_bin, angle_bin] = class_id
            else:

                s = int((self.geometry['size']-1)/2)
                r_lin = np.linspace(self.geometry['resolution'][0]*s, -self.geometry['resolution'][0]*s,
                                    self.geometry['size'])*4
                a_lin = np.linspace(self.geometry['resolution'][1]*s, -self.geometry['resolution'][1]*s,
                                    self.geometry['size'])*4
                
                px_a, px_r = np.meshgrid(a_lin, r_lin)

                if(angle_bin>=s and angle_bin<(self.OUTPUT_DIM[2]-s)):
                    map[0,range_bin-s:range_bin+(s+1),angle_bin-s:angle_bin+(s+1)] = 1
                    map[1,range_bin-s:range_bin+(s+1),angle_bin-s:angle_bin+(s+1)] = ((px_r+range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2,range_bin-s:range_bin+(s+1),angle_bin-s:angle_bin+(s+1)] = ((px_a + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1] 
                    map[3,range_bin-s:range_bin+(s+1),angle_bin-s:angle_bin+(s+1)] = class_id
                elif(angle_bin<s):
                    map[0,range_bin-s:range_bin+(s+1),0:angle_bin+(s+1)] = 1
                    map[1,range_bin-s:range_bin+(s+1),0:angle_bin+(s+1)] = ((px_r[:,s-angle_bin:] + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2,range_bin-s:range_bin+(s+1),0:angle_bin+(s+1)] = ((px_a[:,s-angle_bin:] + angle_mod)- self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
                    map[3,range_bin-s:range_bin+(s+1),0:angle_bin+(s+1)] = class_id
                    
                elif(angle_bin>=self.OUTPUT_DIM[2]):
                    end = s+(self.OUTPUT_DIM[2]-angle_bin)
                    map[0,range_bin-s:range_bin+(s+1),angle_bin-s:] = 1
                    map[1,range_bin-s:range_bin+(s+1),angle_bin-s:] = ((px_r[:,:end] + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2,range_bin-s:range_bin+(s+1),angle_bin-s:] = ((px_a[:,:end] + angle_mod)- self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
                    map[3,range_bin-s:range_bin+(s+1),angle_bin-s:] = class_id

        return map
        
    def decode(self,map,threshold):
       
        range_bins,angle_bins = np.where(map[0,:,:]>=threshold)

        coordinates = []

        for range_bin,angle_bin in zip(range_bins,angle_bins):
            R = range_bin*4*self.geometry['resolution'][0] + map[1,range_bin,angle_bin] * self.statistics['reg_std'][0] + self.statistics['reg_mean'][0]
            A = (angle_bin-self.OUTPUT_DIM[2]/2)*4*self.geometry['resolution'][1] + map[2,range_bin,angle_bin] * self.statistics['reg_std'][1] + self.statistics['reg_mean'][1]
            C = map[0,range_bin,angle_bin]
            CLS = int(np.argmax(map[3:, range_bin, angle_bin]))
        
            coordinates.append([R,A,C,CLS])
       
        return coordinates

class rd_encoder():
    def __init__(self, geometry, statistics,regression_layer = 2):
        
        self.geometry = geometry # Radar dimensions and resolution
        self.statistics = statistics
        self.regression_layer = regression_layer

        # [512, 256]
        self.INPUT_DIM = (geometry['ranges'][0],geometry['ranges'][1])
        self.OUTPUT_DIM = (regression_layer + 2, self.INPUT_DIM[0] // 2, self.INPUT_DIM[1] // 2)

    def encode(self,labels):
        map = np.zeros(self.OUTPUT_DIM)

        for lab in labels:
            # [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]

            if(lab[0]==-1):
                continue

            range_bin = int(np.clip(lab[0] / self.geometry['resolution'][0]/2, 0, self.OUTPUT_DIM[1])) # Convert meter to number of bins. Clip to avoid out of range values.
            range_mod = lab[0] - range_bin * self.geometry['resolution'][0]*2 # Residual value for range regression

            lab_doppler_ms = (lab[2] - self.INPUT_DIM[1] / 2) * self.geometry['resolution'][1] # Convert doppler bin to m/s
            doppler_bin = int(np.clip(np.floor(lab_doppler_ms / self.geometry['resolution'][1] / 2 + self.OUTPUT_DIM[2]/2), 0, self.OUTPUT_DIM[2]))
            doppler_mod = lab_doppler_ms - (doppler_bin - self.OUTPUT_DIM[2]/2) * self.geometry['resolution'][1] * 2 # Residual value for doppler regression

            class_id = lab[10]

            if(self.geometry['size']==1):
                map[0,range_bin,doppler_bin] = 1
                map[1,range_bin,doppler_bin] = (range_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                map[2,range_bin,doppler_bin] = (doppler_mod - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
                map[3, range_bin, doppler_bin] = class_id
            else:
                # Build 2D grid of sub pixel offsets of each detected object
                s = int((self.geometry['size']-1)/2)
                r_lin = np.linspace(self.geometry['resolution'][0]*s, -self.geometry['resolution'][0]*s,
                                    self.geometry['size']) * 2
                d_lin = np.linspace(self.geometry['resolution'][1]*s, -self.geometry['resolution'][1]*s,
                                    self.geometry['size']) * 2
                
                px_a, px_r = np.meshgrid(d_lin, r_lin)

                if(doppler_bin>=s and doppler_bin<(self.OUTPUT_DIM[2]-s)):
                    map[0,range_bin-s:range_bin+(s+1),doppler_bin-s:doppler_bin+(s+1)] = 1
                    map[1,range_bin-s:range_bin+(s+1),doppler_bin-s:doppler_bin+(s+1)] = ((px_r+range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2,range_bin-s:range_bin+(s+1),doppler_bin-s:doppler_bin+(s+1)] = ((px_a + doppler_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1] 
                    map[3,range_bin-s:range_bin+(s+1),doppler_bin-s:doppler_bin+(s+1)] = class_id
                elif(doppler_bin<s):
                    map[0,range_bin-s:range_bin+(s+1),0:doppler_bin+(s+1)] = 1
                    map[1,range_bin-s:range_bin+(s+1),0:doppler_bin+(s+1)] = ((px_r[:,s-doppler_bin:] + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2,range_bin-s:range_bin+(s+1),0:doppler_bin+(s+1)] = ((px_a[:,s-doppler_bin:] + doppler_mod)- self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
                    map[3,range_bin-s:range_bin+(s+1),0:doppler_bin+(s+1)] = class_id

                elif(doppler_bin>=self.OUTPUT_DIM[2]):
                    end = s+(self.OUTPUT_DIM[2]-doppler_bin)
                    map[0,range_bin-s:range_bin+(s+1),doppler_bin-s:] = 1
                    map[1,range_bin-s:range_bin+(s+1),doppler_bin-s:] = ((px_r[:,:end] + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2,range_bin-s:range_bin+(s+1),doppler_bin-s:] = ((px_a[:,:end] + doppler_mod)- self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
                    map[3,range_bin-s:range_bin+(s+1),doppler_bin-s:] = class_id

        return map
        
    def decode(self,map,threshold):
       
        range_bins,doppler_bins = np.where(map[0,:,:]>=threshold)

        coordinates = []

        for range_bin,doppler_bin in zip(range_bins,doppler_bins):
            R = range_bin * 2 * self.geometry['resolution'][0] + map[1,range_bin,doppler_bin] * self.statistics['reg_std'][0] + self.statistics['reg_mean'][0]
            D = (doppler_bin - self.OUTPUT_DIM[2]/2) * 2 * self.geometry['resolution'][1] + map[2,range_bin,doppler_bin] * self.statistics['reg_std'][1] + self.statistics['reg_mean'][1]
            C = map[0,range_bin,doppler_bin]
            CLS = int(np.argmax(map[3:, range_bin, doppler_bin]))

            coordinates.append([R,D,C,CLS])
       
        return coordinates