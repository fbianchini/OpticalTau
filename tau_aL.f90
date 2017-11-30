module tau_aL

  implicit none

  interface
    subroutine RC3JJ(l1, l2, m1, m2, l3min, l3max, wig, ndim, ier)
      real(4), intent(in)  :: l1, l2, m1, m2
      real(4), intent(out) :: l3min, l3max
      integer, intent(in) :: ndim
      real(4), dimension(*), intent(out) :: wig
      integer :: ier
    end subroutine RC3JJ
 end interface

contains 

  subroutine get_AL_tau(cltt, nltt, lmax, AL)
    integer, parameter  :: dp = 4
    integer, intent(in) :: lmax
    real(dp), intent(in), dimension(0:lmax)  :: cltt
    real(dp), intent(in), dimension(0:lmax)  :: nltt
    real(dp), intent(out), dimension(0:lmax) :: AL

    real(dp), parameter  :: one_fourpi = 1.d0 / (atan(1.) * 16.d0)
    integer :: l1, l2, l3min, l3max, dim, ier, l, lwig
    real(dp) :: rl1, rl2, rl3min, rl3max, numb
    real(dp), dimension(0:2*lmax) :: wigner
    real(dp), dimension(0:lmax)   :: ell
    
    wigner = 0.

    ell = [(l, l=0, lmax)]

    do l1=0, lmax
      rl1 = real(l1, kind=dp)
      numb = 0.
       do l2=0, lmax

          fact = (2 * ell(l1) + 1) * (2 * ell(l2) + 1) / (16. * 3.1415)
          fact = fact * cltt(l1) * cltt(l1)
          fact = fact / (cltt(l1) + nltt(l1))
          fact = fact / (cltt(l2) + nltt(l2)) 

          rl2 = real(l2, kind=dp)
          dim = l1 + l2 + 1
          call RC3JJ(rl1, rl2, 0., 0., rl3min, rl3max, wigner, dim, ier)
          if (ier .ne. 0) then
             write(*,*), '!!! ERROR -- Wigner 3J !!!', ier
             stop
          endif
          l3min = int(rl3min)
          l3max = min(int(rl3max), lmax)
          
          do l = l3min, l3max
             lwig = l - l3min 
             numb = numb  +  wigner(lwig) * wigner(lwig)
          enddo

          numb = numb * fact

       enddo
    enddo

  end subroutine get_AL_tau

end module tau_aL