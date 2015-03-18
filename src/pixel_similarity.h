/*
 * pixel_similarity.h
 *
 *  Created on: Mar 5, 2015
 *      Author: ameya
 */

#ifndef PIXEL_SIMILARITY_H_
#define PIXEL_SIMILARITY_H_
int isequal(int r1, int g1, int b1, int r2, int g2, int b2)
{
	if(((r1-r2)*(r1-r2)+(g1-g2)*(g1-g2)+(b1-b2)*(b1-b2))<4)
	{
		return 1;
	}
	else
		return 0;
}




#endif /* PIXEL_SIMILARITY_H_ */
