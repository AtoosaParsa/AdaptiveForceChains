function S=PointForceHPrt(cr,x,y,P,Q)
% Calculate the stress on a half-plane with a normal
% force P and tagential force Q at at position x,y.
% Usage: S=PointForce(r,x,y,P,Q)=>S(:,:,[1 2 3])=[Sr St Srt]
%

% revision history:
% 02/03/93 Mark D. Shattuck <mds> PointForce.m  
%          mds convert from from pointforce.c
% 01/30/06 mds added abs(cr)
% 10/26/08 mds modify for PointForce function
% 11/15/14 mds redo for book

if(~exist('Q','var') || isempty(Q))
  Q=0;
end

[Nx Ny]=size(cr);
S=zeros(Nx,Ny,3);
cr=cr-x-1i*y;

r=abs(cr);
th=angle(cr);

S(:,:,1)=(-2/pi)*(P*sin(th)+Q*cos(th))./r;



